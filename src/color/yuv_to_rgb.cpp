// SPDX-License-Identifier: Apache-2.0

#include "yuv_to_rgb.h"

#include <chrono>
#include <cstdint>
#include <malloc.h>

#include <windows.h>

#include <cuda_runtime.h>
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

// Page-locked (pinned) host allocation for the buffers uploaded to the device. Pinned memory lets
// cudaMemcpyAsync run as a true async DMA (pageable memory forces a synchronous staging copy), and
// CUDA returns page-aligned memory, which already satisfies zimg's 64-byte base alignment.
auto PinnedAlloc(size_t bytes) -> void * {
    void *p = nullptr;
    if (cudaHostAlloc(&p, RoundUp(bytes, kAlign), cudaHostAllocDefault) != cudaSuccess) {
        return nullptr;
    }
    return p;
}

struct FormatInfo {
    int subsampleW; // log2
    int subsampleH; // log2
    int depth;      // bits per component (8, 10, 16)
};

auto InfoFor(YuvToRgbConverter::Kind kind) -> FormatInfo {
    using K = YuvToRgbConverter::Kind;
    switch (kind) {
    case K::NV12: return { 1, 1, 8 };
    case K::P010: return { 1, 1, 10 };
    }
    return { 1, 1, 8 };
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

    const int cW = params.width >> info.subsampleW;
    const int cH = params.height >> info.subsampleH;
    const size_t bps = converter->_planeBytes;

    converter->_stride0 = static_cast<ptrdiff_t>(RoundUp(static_cast<size_t>(params.width) * bps, kAlign));
    converter->_strideC = static_cast<ptrdiff_t>(RoundUp(static_cast<size_t>(cW) * bps, kAlign));
    converter->_strideRgb = static_cast<ptrdiff_t>(RoundUp(static_cast<size_t>(params.width) * sizeof(unsigned short), kAlign));

    converter->_p0 = AlignedAlloc(converter->_stride0 * params.height);
    converter->_p1 = AlignedAlloc(converter->_strideC * cH);
    converter->_p2 = AlignedAlloc(converter->_strideC * cH);
    converter->_r = static_cast<unsigned short *>(PinnedAlloc(converter->_strideRgb * params.height));
    converter->_g = static_cast<unsigned short *>(PinnedAlloc(converter->_strideRgb * params.height));
    converter->_b = static_cast<unsigned short *>(PinnedAlloc(converter->_strideRgb * params.height));

    if (!converter->_p0 || !converter->_p1 || !converter->_p2 || !converter->_r || !converter->_g || !converter->_b) {
        error = L"Out of memory allocating conversion buffers.";
        return nullptr;
    }

    converter->_result.r = converter->_r;
    converter->_result.g = converter->_g;
    converter->_result.b = converter->_b;
    converter->_result.strideBytes = converter->_strideRgb;

    zimg_image_format src;
    zimg_image_format dst;
    zimg_image_format_default(&src, ZIMG_API_VERSION);
    zimg_image_format_default(&dst, ZIMG_API_VERSION);

    src.width = static_cast<unsigned>(params.width);
    src.height = static_cast<unsigned>(params.height);
    src.pixel_type = converter->_planeBytes == 1 ? ZIMG_PIXEL_BYTE : ZIMG_PIXEL_WORD;
    src.subsample_w = static_cast<unsigned>(info.subsampleW);
    src.subsample_h = static_cast<unsigned>(info.subsampleH);
    src.color_family = ZIMG_COLOR_YUV;
    src.matrix_coefficients = params.bt709 ? ZIMG_MATRIX_BT709 : ZIMG_MATRIX_BT470_BG;
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
    cudaFreeHost(_r);
    cudaFreeHost(_g);
    cudaFreeHost(_b);
    _aligned_free(_tmp);
}

// Splits the semi-planar source into the planar buffers zimg consumes. Plane 0 = Y, plane 1 = U,
// and plane 2 = V. P010 samples are shifted right to convert from MSB-aligned to 10-bit values.
auto YuvToRgbConverter::Deinterleave(const unsigned char *srcBuffer) -> void {
    const int w = _width;
    const int h = _height;
    const int cW = w >> _subsampleW;
    const int cH = h >> _subsampleH;
    const int bps = _planeBytes;

    switch (_kind) {
    case Kind::NV12:
    case Kind::P010: {
        const ptrdiff_t sY = static_cast<ptrdiff_t>(w) * bps;
        const unsigned char *srcUV = srcBuffer + sY * h;
        const int shift = _kind == Kind::P010 ? 6 : 0;
        if (bps == 1) {
            CopyPlane<uint8_t>(_p0, _stride0, srcBuffer, sY, w, h, 0);
            SplitUV<uint8_t>(_p1, _p2, _strideC, srcUV, sY, cW, cH, 0);
        } else {
            CopyPlane<uint16_t>(_p0, _stride0, srcBuffer, sY, w, h, shift);
            SplitUV<uint16_t>(_p1, _p2, _strideC, srcUV, sY, cW, cH, shift);
        }
        break;
    }

    }
}

auto YuvToRgbConverter::Convert(const unsigned char *srcBuffer, StageTimings *timings) -> const PlanarRgbFp16 * {
    // Only consult the clock when a breakdown was requested; the production path passes nullptr.
    using Clock = std::chrono::steady_clock;
    const auto now = [&] { return Clock::now(); };
    const auto elapsed = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };

    const auto t0 = timings ? now() : Clock::time_point{};
    Deinterleave(srcBuffer);
    const auto t1 = timings ? now() : Clock::time_point{};

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
    const auto t2 = timings ? now() : Clock::time_point{};

    if (timings) {
        timings->deinterleaveSec = elapsed(t0, t1);
        timings->zimgSec = elapsed(t1, t2);
    }
    return &_result;
}

}
