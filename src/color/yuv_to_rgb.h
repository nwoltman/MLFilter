// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

struct zimg_filter_graph;

namespace MLFilter {

// Converts a decoded video frame (any of the supported YUV/RGB layouts) to planar, full-range
// RGB fp16 using zimg — the same library VapourSynth's resize.Bicubic uses — so the result
// matches the reference python pipeline: Catmull-Rom (b=0, c=0.5) chroma upsampling, the
// source color matrix and range, left chroma siting, then a clamp to [0,1].
//
// zimg does all of the conversion (subsampling, depth, matrix, range, RGB); this class only
// unpacks each format's memory layout into the planar buffers zimg consumes. The output is 3
// contiguous planes (R, then G, then B), each width*height fp16 with stride == width — exactly
// the NCHW buffer the engine consumes. One converter per pin connection; not thread-safe.
class YuvToRgbConverter {
public:
    enum class Kind {
        // 4:2:0
        NV12, // 8-bit, Y + interleaved UV
        YV12, // 8-bit, planar Y,V,U
        P010, // 10-bit (value in high bits), Y + interleaved UV
        P016, // 16-bit, Y + interleaved UV
        // 4:2:2
        YUY2, // 8-bit packed Y0 U Y1 V
        UYVY, // 8-bit packed U Y0 V Y1
        P210, // 10-bit (value in high bits), Y + interleaved UV
        P216, // 16-bit, Y + interleaved UV
        V210, // 10-bit packed (6 samples per 16 bytes)
        // 4:4:4
        YV24, // 8-bit planar Y,V,U
        AYUV, // 8-bit packed (V,U,Y,A per pixel)
        Y410, // 10-bit packed in a DWORD (U:10 Y:10 V:10 A:2)
        V410, // 10-bit packed in a DWORD (pad:2 U:10 Y:10 V:10)
        Y416, // 16-bit packed (U,Y,V,A per pixel)
        // RGB
        RGB24, // 8-bit packed B,G,R (bottom-up DIB by default)
        RGB32, // 8-bit packed B,G,R,X
        RGB48, // 16-bit packed R,G,B
    };

    struct Params {
        Kind kind;
        int width;       // == engine input width (the connected video resolution)
        int height;
        bool bt709;      // YUV matrix: true = BT.709, false = BT.601/SMPTE-170M (ignored for RGB)
        bool fullRange;  // input pixel range: true = full, false = limited
        bool bottomUp;   // RGB stored bottom-up (positive biHeight); ignored for YUV
    };

    static auto Create(const Params &params, std::wstring &error) -> std::unique_ptr<YuvToRgbConverter>;

    ~YuvToRgbConverter();

    YuvToRgbConverter(const YuvToRgbConverter &) = delete;
    auto operator=(const YuvToRgbConverter &) -> YuvToRgbConverter & = delete;

    // Converts one frame. srcBuffer points at the IMediaSample data. Returns a pointer to the
    // planar RGB fp16 result (size OutputBytes()), valid until the next Convert() or
    // destruction. Returns nullptr on error.
    auto Convert(const unsigned char *srcBuffer) -> const unsigned short *;

    auto OutputBytes() const -> size_t { return static_cast<size_t>(3) * _width * _height * sizeof(unsigned short); }

private:
    YuvToRgbConverter() = default;

    auto Deinterleave(const unsigned char *srcBuffer) -> void;
    auto ClampAndPack() -> void;

    zimg_filter_graph *_graph = nullptr;

    Kind _kind = Kind::NV12;
    int _width = 0;
    int _height = 0;
    int _subsampleW = 0; // log2 horizontal chroma subsampling
    int _subsampleH = 0; // log2 vertical chroma subsampling
    int _planeBytes = 1; // bytes per planar sample fed to zimg (1 for 8-bit, 2 for 10/16-bit)
    bool _rgb = false;
    bool _bottomUp = false;

    // Aligned planar input buffers fed to zimg: plane 0 = Y (or R), 1 = U (or G), 2 = V (or B).
    void *_p0 = nullptr;
    void *_p1 = nullptr;
    void *_p2 = nullptr;
    ptrdiff_t _stride0 = 0; // bytes
    ptrdiff_t _strideC = 0; // bytes (planes 1 and 2)

    // Aligned planar RGB fp16 output from zimg (padded stride), then clamped/repacked tightly.
    unsigned short *_r = nullptr;
    unsigned short *_g = nullptr;
    unsigned short *_b = nullptr;
    ptrdiff_t _strideRgb = 0; // bytes

    void *_tmp = nullptr; // zimg scratch buffer

    unsigned short *_out = nullptr; // tight NCHW result returned to the caller
};

}
