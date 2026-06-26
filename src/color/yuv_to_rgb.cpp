// SPDX-License-Identifier: Apache-2.0

#include "yuv_to_rgb.h"

#include <cstdint>
#include <malloc.h>

#include <windows.h>

#include <zimg.h>

namespace MLFilter {

namespace {

// zimg requires plane base addresses and strides to be a multiple of the host alignment
// (32 bytes on x86; 64 covers the 64-byte/AVX-512 path too).
constexpr size_t kAlign = 64;

auto RoundUp(size_t value, size_t multiple) -> size_t {
    return (value + multiple - 1) / multiple * multiple;
}

auto AlignedAlloc(size_t bytes) -> void * {
    return _aligned_malloc(RoundUp(bytes, kAlign), kAlign);
}

auto Align4(int value) -> int {
    return (value + 3) & ~3;
}

// fp16 bit patterns: clamp a half to [0,1] purely on its 16-bit representation. Non-negative
// halves are monotonic in their bit pattern, so 0x3C00 (== 1.0) is the upper bound; anything
// with the sign bit set is negative -> 0. (+inf/+nan, being > 0x3C00, clamp to 1.0.)
constexpr uint16_t kHalfOne = 0x3C00;

auto ClampHalf(uint16_t h) -> uint16_t {
    if (h & 0x8000u) {
        return 0;
    }
    return h > kHalfOne ? kHalfOne : h;
}

struct FormatInfo {
    int subsampleW; // log2
    int subsampleH; // log2
    int depth;      // bits per component (8, 10, 16)
    bool rgb;
};

auto InfoFor(YuvToRgbConverter::Kind kind) -> FormatInfo {
    using K = YuvToRgbConverter::Kind;
    switch (kind) {
    case K::NV12: return { 1, 1, 8, false };
    case K::YV12: return { 1, 1, 8, false };
    case K::P010: return { 1, 1, 10, false };
    case K::P016: return { 1, 1, 16, false };
    case K::YUY2: return { 1, 0, 8, false };
    case K::UYVY: return { 1, 0, 8, false };
    case K::P210: return { 1, 0, 10, false };
    case K::P216: return { 1, 0, 16, false };
    case K::V210: return { 1, 0, 10, false };
    case K::YV24: return { 0, 0, 8, false };
    case K::AYUV: return { 0, 0, 8, false };
    case K::Y410: return { 0, 0, 10, false };
    case K::V410: return { 0, 0, 10, false };
    case K::Y416: return { 0, 0, 16, false };
    case K::RGB24: return { 0, 0, 8, true };
    case K::RGB32: return { 0, 0, 8, true };
    case K::RGB48: return { 0, 0, 16, true };
    }
    return { 1, 1, 8, false };
}

// Copies one plane (full or chroma), right-shifting each sample by `shift` (for 10-bit-in-16
// formats whose value is left-justified). T is the planar sample type (uint8_t / uint16_t).
template <typename T>
auto CopyPlane(void *dst, ptrdiff_t dstStride, const unsigned char *src, ptrdiff_t srcStride, int w, int h, int shift) -> void {
    for (int y = 0; y < h; ++y) {
        const auto *s = reinterpret_cast<const T *>(src + static_cast<ptrdiff_t>(y) * srcStride);
        auto *d = reinterpret_cast<T *>(static_cast<unsigned char *>(dst) + static_cast<ptrdiff_t>(y) * dstStride);
        for (int x = 0; x < w; ++x) {
            d[x] = static_cast<T>(s[x] >> shift);
        }
    }
}

// Splits an interleaved chroma plane (U,V,U,V,...) into separate U and V planes.
template <typename T>
auto SplitUV(void *uDst, void *vDst, ptrdiff_t dstStride, const unsigned char *src, ptrdiff_t srcStride, int cW, int cH, int shift) -> void {
    for (int y = 0; y < cH; ++y) {
        const auto *s = reinterpret_cast<const T *>(src + static_cast<ptrdiff_t>(y) * srcStride);
        auto *u = reinterpret_cast<T *>(static_cast<unsigned char *>(uDst) + static_cast<ptrdiff_t>(y) * dstStride);
        auto *v = reinterpret_cast<T *>(static_cast<unsigned char *>(vDst) + static_cast<ptrdiff_t>(y) * dstStride);
        for (int x = 0; x < cW; ++x) {
            u[x] = static_cast<T>(s[2 * x + 0] >> shift);
            v[x] = static_cast<T>(s[2 * x + 1] >> shift);
        }
    }
}

}

auto YuvToRgbConverter::Create(const Params &params, std::wstring &error) -> std::unique_ptr<YuvToRgbConverter> {
    const FormatInfo info = InfoFor(params.kind);

    if (params.width <= 0 || params.height <= 0 ||
        ((params.width & 1) && info.subsampleW) || ((params.height & 1) && info.subsampleH)) {
        error = L"Invalid (or non-even subsampled) video dimensions for conversion.";
        return nullptr;
    }

    auto converter = std::unique_ptr<YuvToRgbConverter>(new YuvToRgbConverter());
    converter->_kind = params.kind;
    converter->_width = params.width;
    converter->_height = params.height;
    converter->_subsampleW = info.subsampleW;
    converter->_subsampleH = info.subsampleH;
    converter->_planeBytes = info.depth <= 8 ? 1 : 2;
    converter->_rgb = info.rgb;
    converter->_bottomUp = params.bottomUp;

    const int cW = params.width >> info.subsampleW;
    const int cH = params.height >> info.subsampleH;
    const size_t bps = converter->_planeBytes;

    converter->_stride0 = static_cast<ptrdiff_t>(RoundUp(static_cast<size_t>(params.width) * bps, kAlign));
    converter->_strideC = static_cast<ptrdiff_t>(RoundUp(static_cast<size_t>(cW) * bps, kAlign));
    converter->_strideRgb = static_cast<ptrdiff_t>(RoundUp(static_cast<size_t>(params.width) * sizeof(unsigned short), kAlign));

    converter->_p0 = AlignedAlloc(converter->_stride0 * params.height);
    converter->_p1 = AlignedAlloc(converter->_strideC * cH);
    converter->_p2 = AlignedAlloc(converter->_strideC * cH);
    converter->_r = static_cast<unsigned short *>(AlignedAlloc(converter->_strideRgb * params.height));
    converter->_g = static_cast<unsigned short *>(AlignedAlloc(converter->_strideRgb * params.height));
    converter->_b = static_cast<unsigned short *>(AlignedAlloc(converter->_strideRgb * params.height));
    converter->_out = static_cast<unsigned short *>(_aligned_malloc(converter->OutputBytes(), kAlign));

    if (!converter->_p0 || !converter->_p1 || !converter->_p2 || !converter->_r || !converter->_g || !converter->_b || !converter->_out) {
        error = L"Out of memory allocating conversion buffers.";
        return nullptr;
    }

    zimg_image_format src;
    zimg_image_format dst;
    zimg_image_format_default(&src, ZIMG_API_VERSION);
    zimg_image_format_default(&dst, ZIMG_API_VERSION);

    src.width = static_cast<unsigned>(params.width);
    src.height = static_cast<unsigned>(params.height);
    src.pixel_type = converter->_planeBytes == 1 ? ZIMG_PIXEL_BYTE : ZIMG_PIXEL_WORD;
    src.subsample_w = static_cast<unsigned>(info.subsampleW);
    src.subsample_h = static_cast<unsigned>(info.subsampleH);
    src.color_family = info.rgb ? ZIMG_COLOR_RGB : ZIMG_COLOR_YUV;
    src.matrix_coefficients = info.rgb ? ZIMG_MATRIX_RGB : (params.bt709 ? ZIMG_MATRIX_BT709 : ZIMG_MATRIX_BT470_BG);
    src.transfer_characteristics = ZIMG_TRANSFER_UNSPECIFIED; // no transfer conversion
    src.color_primaries = ZIMG_PRIMARIES_UNSPECIFIED;         // no gamut conversion
    src.depth = static_cast<unsigned>(info.depth);
    src.pixel_range = params.fullRange ? ZIMG_RANGE_FULL : ZIMG_RANGE_LIMITED;
    src.chroma_location = ZIMG_CHROMA_LEFT;

    dst.width = static_cast<unsigned>(params.width);
    dst.height = static_cast<unsigned>(params.height);
    dst.pixel_type = ZIMG_PIXEL_HALF;
    dst.color_family = ZIMG_COLOR_RGB;
    dst.matrix_coefficients = ZIMG_MATRIX_RGB;
    dst.transfer_characteristics = ZIMG_TRANSFER_UNSPECIFIED;
    dst.color_primaries = ZIMG_PRIMARIES_UNSPECIFIED;
    dst.depth = 16;
    dst.pixel_range = ZIMG_RANGE_FULL;

    zimg_graph_builder_params gparams;
    zimg_graph_builder_params_default(&gparams, ZIMG_API_VERSION);
    // Catmull-Rom (b=0, c=0.5) for chroma upsampling, matching resize.Bicubic(filter_param_a=0,
    // filter_param_b=0.5). Luma isn't resized (same resolution) but mirror the params anyway.
    gparams.resample_filter = ZIMG_RESIZE_BICUBIC;
    gparams.filter_param_a = 0.0;
    gparams.filter_param_b = 0.5;
    gparams.resample_filter_uv = ZIMG_RESIZE_BICUBIC;
    gparams.filter_param_a_uv = 0.0;
    gparams.filter_param_b_uv = 0.5;
    gparams.dither_type = ZIMG_DITHER_NONE;

    converter->_graph = zimg_filter_graph_build(&src, &dst, &gparams);
    if (converter->_graph == nullptr) {
        char msg[256] = {};
        zimg_get_last_error(msg, sizeof(msg));
        const int needed = MultiByteToWideChar(CP_UTF8, 0, msg, -1, nullptr, 0);
        std::wstring wide(needed > 0 ? static_cast<size_t>(needed - 1) : 0, L'\0');
        if (needed > 0) {
            MultiByteToWideChar(CP_UTF8, 0, msg, -1, wide.data(), needed);
        }
        error = L"zimg could not build the conversion graph: " + wide;
        return nullptr;
    }

    size_t tmpSize = 0;
    if (zimg_filter_graph_get_tmp_size(converter->_graph, &tmpSize) != ZIMG_ERROR_SUCCESS) {
        error = L"zimg failed to report its scratch buffer size.";
        return nullptr;
    }
    converter->_tmp = _aligned_malloc(RoundUp(tmpSize, kAlign), kAlign);
    if (converter->_tmp == nullptr) {
        error = L"Out of memory allocating the zimg scratch buffer.";
        return nullptr;
    }

    return converter;
}

YuvToRgbConverter::~YuvToRgbConverter() {
    if (_graph != nullptr) {
        zimg_filter_graph_free(_graph);
    }
    _aligned_free(_p0);
    _aligned_free(_p1);
    _aligned_free(_p2);
    _aligned_free(_r);
    _aligned_free(_g);
    _aligned_free(_b);
    _aligned_free(_tmp);
    _aligned_free(_out);
}

// Unpacks the source frame's layout into the planar buffers zimg consumes. Plane 0 = Y (or R),
// plane 1 = U (or G), plane 2 = V (or B). 10-bit-in-16 formats (P0xx/P2xx) are right-shifted to
// right-justify the value; bit-packed formats (v210/Y410/v410) extract the 10-bit fields.
auto YuvToRgbConverter::Deinterleave(const unsigned char *srcBuffer) -> void {
    const int w = _width;
    const int h = _height;
    const int cW = w >> _subsampleW;
    const int cH = h >> _subsampleH;
    const int bps = _planeBytes;

    switch (_kind) {
    case Kind::NV12:
    case Kind::P210:
    case Kind::P010:
    case Kind::P016:
    case Kind::P216: {
        const ptrdiff_t sY = static_cast<ptrdiff_t>(w) * bps;
        const unsigned char *srcUV = srcBuffer + sY * h;
        const int shift = (_kind == Kind::P010 || _kind == Kind::P210) ? 6 : 0;
        if (bps == 1) {
            CopyPlane<uint8_t>(_p0, _stride0, srcBuffer, sY, w, h, 0);
            SplitUV<uint8_t>(_p1, _p2, _strideC, srcUV, sY, cW, cH, 0);
        } else {
            CopyPlane<uint16_t>(_p0, _stride0, srcBuffer, sY, w, h, shift);
            SplitUV<uint16_t>(_p1, _p2, _strideC, srcUV, sY, cW, cH, shift);
        }
        break;
    }

    case Kind::YV12:
    case Kind::YV24: {
        // Planar, plane order Y, V, U.
        const ptrdiff_t sY = static_cast<ptrdiff_t>(w) * bps;
        const ptrdiff_t sC = static_cast<ptrdiff_t>(cW) * bps;
        const unsigned char *srcV = srcBuffer + sY * h;
        const unsigned char *srcU = srcV + sC * cH;
        CopyPlane<uint8_t>(_p0, _stride0, srcBuffer, sY, w, h, 0);
        CopyPlane<uint8_t>(_p2, _strideC, srcV, sC, cW, cH, 0); // V -> plane 2
        CopyPlane<uint8_t>(_p1, _strideC, srcU, sC, cW, cH, 0); // U -> plane 1
        break;
    }

    case Kind::YUY2:
    case Kind::UYVY: {
        const ptrdiff_t srcStride = static_cast<ptrdiff_t>(w) * 2;
        const int yOff = _kind == Kind::YUY2 ? 0 : 1; // YUY2: Y U Y V ; UYVY: U Y V Y
        const int cOff = _kind == Kind::YUY2 ? 1 : 0;
        for (int y = 0; y < h; ++y) {
            const uint8_t *s = srcBuffer + static_cast<ptrdiff_t>(y) * srcStride;
            auto *yr = static_cast<uint8_t *>(_p0) + static_cast<ptrdiff_t>(y) * _stride0;
            auto *ur = static_cast<uint8_t *>(_p1) + static_cast<ptrdiff_t>(y) * _strideC;
            auto *vr = static_cast<uint8_t *>(_p2) + static_cast<ptrdiff_t>(y) * _strideC;
            for (int x = 0; x < cW; ++x) {
                yr[2 * x + 0] = s[4 * x + yOff];
                yr[2 * x + 1] = s[4 * x + yOff + 2];
                ur[x] = s[4 * x + cOff];
                vr[x] = s[4 * x + cOff + 2];
            }
        }
        break;
    }

    case Kind::V210: {
        // 10-bit 4:2:2 packed: 4 little-endian DWORDs hold 6 luma + 3 chroma pairs.
        const ptrdiff_t srcStride = (static_cast<ptrdiff_t>(w) + 47) / 48 * 128;
        const int groups = (w + 5) / 6;
        for (int y = 0; y < h; ++y) {
            const auto *words = reinterpret_cast<const uint32_t *>(srcBuffer + static_cast<ptrdiff_t>(y) * srcStride);
            auto *yr = reinterpret_cast<uint16_t *>(static_cast<unsigned char *>(_p0) + static_cast<ptrdiff_t>(y) * _stride0);
            auto *ur = reinterpret_cast<uint16_t *>(static_cast<unsigned char *>(_p1) + static_cast<ptrdiff_t>(y) * _strideC);
            auto *vr = reinterpret_cast<uint16_t *>(static_cast<unsigned char *>(_p2) + static_cast<ptrdiff_t>(y) * _strideC);
            for (int g = 0; g < groups; ++g) {
                const uint32_t w0 = words[4 * g + 0];
                const uint32_t w1 = words[4 * g + 1];
                const uint32_t w2 = words[4 * g + 2];
                const uint32_t w3 = words[4 * g + 3];
                const uint16_t U0 = w0 & 0x3FF, Y0 = (w0 >> 10) & 0x3FF, V0 = (w0 >> 20) & 0x3FF;
                const uint16_t Y1 = w1 & 0x3FF, U2 = (w1 >> 10) & 0x3FF, Y2 = (w1 >> 20) & 0x3FF;
                const uint16_t V2 = w2 & 0x3FF, Y3 = (w2 >> 10) & 0x3FF, U4 = (w2 >> 20) & 0x3FF;
                const uint16_t Y4 = w3 & 0x3FF, V4 = (w3 >> 10) & 0x3FF, Y5 = (w3 >> 20) & 0x3FF;
                const int yb = 6 * g;
                const int cb = 3 * g;
                const uint16_t lum[6] = { Y0, Y1, Y2, Y3, Y4, Y5 };
                for (int i = 0; i < 6; ++i) {
                    if (yb + i < w) {
                        yr[yb + i] = lum[i];
                    }
                }
                const uint16_t us[3] = { U0, U2, U4 };
                const uint16_t vs[3] = { V0, V2, V4 };
                for (int i = 0; i < 3; ++i) {
                    if (cb + i < cW) {
                        ur[cb + i] = us[i];
                        vr[cb + i] = vs[i];
                    }
                }
            }
        }
        break;
    }

    case Kind::AYUV: {
        // 8-bit 4:4:4 packed, bytes per pixel: V, U, Y, A.
        const ptrdiff_t srcStride = static_cast<ptrdiff_t>(w) * 4;
        for (int y = 0; y < h; ++y) {
            const uint8_t *s = srcBuffer + static_cast<ptrdiff_t>(y) * srcStride;
            auto *yr = static_cast<uint8_t *>(_p0) + static_cast<ptrdiff_t>(y) * _stride0;
            auto *ur = static_cast<uint8_t *>(_p1) + static_cast<ptrdiff_t>(y) * _strideC;
            auto *vr = static_cast<uint8_t *>(_p2) + static_cast<ptrdiff_t>(y) * _strideC;
            for (int x = 0; x < w; ++x) {
                vr[x] = s[4 * x + 0];
                ur[x] = s[4 * x + 1];
                yr[x] = s[4 * x + 2];
            }
        }
        break;
    }

    case Kind::Y410:
    case Kind::V410: {
        // 10-bit 4:4:4 packed in a DWORD. Y410: U[0:10] Y[10:20] V[20:30] A[30:32].
        //                                 v410: pad[0:2] U[2:12] Y[12:22] V[22:32].
        const ptrdiff_t srcStride = static_cast<ptrdiff_t>(w) * 4;
        const int sh = _kind == Kind::Y410 ? 0 : 2;
        for (int y = 0; y < h; ++y) {
            const auto *s = reinterpret_cast<const uint32_t *>(srcBuffer + static_cast<ptrdiff_t>(y) * srcStride);
            auto *yr = reinterpret_cast<uint16_t *>(static_cast<unsigned char *>(_p0) + static_cast<ptrdiff_t>(y) * _stride0);
            auto *ur = reinterpret_cast<uint16_t *>(static_cast<unsigned char *>(_p1) + static_cast<ptrdiff_t>(y) * _strideC);
            auto *vr = reinterpret_cast<uint16_t *>(static_cast<unsigned char *>(_p2) + static_cast<ptrdiff_t>(y) * _strideC);
            for (int x = 0; x < w; ++x) {
                const uint32_t px = s[x];
                ur[x] = static_cast<uint16_t>((px >> (sh + 0)) & 0x3FF);
                yr[x] = static_cast<uint16_t>((px >> (sh + 10)) & 0x3FF);
                vr[x] = static_cast<uint16_t>((px >> (sh + 20)) & 0x3FF);
            }
        }
        break;
    }

    case Kind::Y416: {
        // 16-bit 4:4:4 packed, uint16 per pixel: U, Y, V, A.
        const ptrdiff_t srcStride = static_cast<ptrdiff_t>(w) * 8;
        for (int y = 0; y < h; ++y) {
            const auto *s = reinterpret_cast<const uint16_t *>(srcBuffer + static_cast<ptrdiff_t>(y) * srcStride);
            auto *yr = reinterpret_cast<uint16_t *>(static_cast<unsigned char *>(_p0) + static_cast<ptrdiff_t>(y) * _stride0);
            auto *ur = reinterpret_cast<uint16_t *>(static_cast<unsigned char *>(_p1) + static_cast<ptrdiff_t>(y) * _strideC);
            auto *vr = reinterpret_cast<uint16_t *>(static_cast<unsigned char *>(_p2) + static_cast<ptrdiff_t>(y) * _strideC);
            for (int x = 0; x < w; ++x) {
                ur[x] = s[4 * x + 0];
                yr[x] = s[4 * x + 1];
                vr[x] = s[4 * x + 2];
            }
        }
        break;
    }

    case Kind::RGB24:
    case Kind::RGB32:
    case Kind::RGB48: {
        // Packed RGB. 24/32-bit are byte order B,G,R(,X); 48-bit is uint16 R,G,B. DIBs are
        // bottom-up when biHeight > 0 (read rows in reverse so the planar output is top-down).
        const int bytesPerPixel = _kind == Kind::RGB24 ? 3 : (_kind == Kind::RGB32 ? 4 : 6);
        const ptrdiff_t srcStride = _kind == Kind::RGB32 ? static_cast<ptrdiff_t>(w) * 4 : Align4(w * bytesPerPixel);
        for (int y = 0; y < h; ++y) {
            const int srcY = _bottomUp ? (h - 1 - y) : y;
            const unsigned char *row = srcBuffer + static_cast<ptrdiff_t>(srcY) * srcStride;
            if (_kind == Kind::RGB48) {
                const auto *s = reinterpret_cast<const uint16_t *>(row);
                auto *rr = reinterpret_cast<uint16_t *>(static_cast<unsigned char *>(_p0) + static_cast<ptrdiff_t>(y) * _stride0);
                auto *gg = reinterpret_cast<uint16_t *>(static_cast<unsigned char *>(_p1) + static_cast<ptrdiff_t>(y) * _strideC);
                auto *bb = reinterpret_cast<uint16_t *>(static_cast<unsigned char *>(_p2) + static_cast<ptrdiff_t>(y) * _strideC);
                for (int x = 0; x < w; ++x) {
                    rr[x] = s[3 * x + 0];
                    gg[x] = s[3 * x + 1];
                    bb[x] = s[3 * x + 2];
                }
            } else {
                auto *rr = static_cast<uint8_t *>(_p0) + static_cast<ptrdiff_t>(y) * _stride0;
                auto *gg = static_cast<uint8_t *>(_p1) + static_cast<ptrdiff_t>(y) * _strideC;
                auto *bb = static_cast<uint8_t *>(_p2) + static_cast<ptrdiff_t>(y) * _strideC;
                for (int x = 0; x < w; ++x) {
                    bb[x] = row[bytesPerPixel * x + 0];
                    gg[x] = row[bytesPerPixel * x + 1];
                    rr[x] = row[bytesPerPixel * x + 2];
                }
            }
        }
        break;
    }
    }
}

// Clamps zimg's (padded-stride) planar RGB fp16 to [0,1] and repacks it tightly into _out as
// 3 contiguous planes (R, G, B), stride == width — the engine's NCHW input layout.
auto YuvToRgbConverter::ClampAndPack() -> void {
    const size_t plane = static_cast<size_t>(_width) * _height;
    const unsigned short *srcPlanes[3] = { _r, _g, _b };

    for (int p = 0; p < 3; ++p) {
        unsigned short *dst = _out + static_cast<size_t>(p) * plane;
        for (int y = 0; y < _height; ++y) {
            const auto *srcRow = reinterpret_cast<const unsigned short *>(
                reinterpret_cast<const unsigned char *>(srcPlanes[p]) + static_cast<ptrdiff_t>(y) * _strideRgb);
            unsigned short *dstRow = dst + static_cast<size_t>(y) * _width;
            for (int x = 0; x < _width; ++x) {
                dstRow[x] = ClampHalf(srcRow[x]);
            }
        }
    }
}

auto YuvToRgbConverter::Convert(const unsigned char *srcBuffer) -> const unsigned short * {
    Deinterleave(srcBuffer);

    zimg_image_buffer_const src = {};
    src.version = ZIMG_API_VERSION;
    src.plane[0].data = _p0;
    src.plane[0].stride = _stride0;
    src.plane[0].mask = ZIMG_BUFFER_MAX;
    src.plane[1].data = _p1;
    src.plane[1].stride = _strideC;
    src.plane[1].mask = ZIMG_BUFFER_MAX;
    src.plane[2].data = _p2;
    src.plane[2].stride = _strideC;
    src.plane[2].mask = ZIMG_BUFFER_MAX;

    zimg_image_buffer dst = {};
    dst.version = ZIMG_API_VERSION;
    dst.plane[0].data = _r;
    dst.plane[0].stride = _strideRgb;
    dst.plane[0].mask = ZIMG_BUFFER_MAX;
    dst.plane[1].data = _g;
    dst.plane[1].stride = _strideRgb;
    dst.plane[1].mask = ZIMG_BUFFER_MAX;
    dst.plane[2].data = _b;
    dst.plane[2].stride = _strideRgb;
    dst.plane[2].mask = ZIMG_BUFFER_MAX;

    if (zimg_filter_graph_process(_graph, &src, &dst, _tmp, nullptr, nullptr, nullptr, nullptr) != ZIMG_ERROR_SUCCESS) {
        return nullptr;
    }

    ClampAndPack();
    return _out;
}

}
