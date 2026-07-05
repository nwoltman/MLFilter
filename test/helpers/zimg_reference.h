// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

struct zimg_filter_graph;

namespace MLFilter {

// Test reference: converts an NV12 or P010 decoded video frame to planar, full-range
// RGB fp16 using zimg with settings: Bicubic Catmull-Rom (b=0, c=0.5) chroma upsampling, the
// source color matrix and range, left chroma siting, then a clamp to [0,1].
//
// zimg does the conversion (subsampling, depth, matrix, range, RGB); this class only splits
// the semi-planar input into the planar buffers zimg consumes. The output is 3
// planar channels (R, G, B), each width*height fp16, in pinned host memory so the engine can
// upload them to the device directly. The [0,1] clamp the model input needs is applied on the
// GPU. One converter per pin connection; not thread-safe.
class YuvToRgbConverter {
public:
    enum class Kind {
        NV12, // 8-bit, Y + interleaved UV
        P010, // 10-bit (value in high bits), Y + interleaved UV
    };

    struct Params {
        Kind kind;
        int width;       // == engine input width (the connected video resolution)
        int height;
        bool bt709;      // YUV matrix: true = BT.709, false = BT.601/SMPTE-170M
        bool fullRange;  // input pixel range: true = full, false = limited
    };

    // Optional per-stage wall-clock breakdown of one Convert() call (seconds). Used by the
    // benchmark to see where the CPU conversion cost goes; nullptr in the production path.
    struct StageTimings {
        double deinterleaveSec = 0;
        double zimgSec = 0;
    };

    // A successful Convert() result: three padded-stride planar fp16 channels (R, G, B), each
    // width*height __half with per-row stride strideBytes, in pinned host memory. NOT yet clamped
    // to [0,1] — the engine applies that on the GPU. Valid until the next Convert() or destruction.
    struct PlanarRgbFp16 {
        const void *r = nullptr;
        const void *g = nullptr;
        const void *b = nullptr;
        ptrdiff_t strideBytes = 0;
    };

    static auto Create(const Params &params, std::wstring &error) -> std::unique_ptr<YuvToRgbConverter>;

    ~YuvToRgbConverter();

    YuvToRgbConverter(const YuvToRgbConverter &) = delete;
    auto operator=(const YuvToRgbConverter &) -> YuvToRgbConverter & = delete;

    // Converts one frame. srcBuffer points at the IMediaSample data. Returns a pointer to the
    // planar RGB fp16 result (owned by the converter), valid until the next Convert() or
    // destruction. Returns nullptr on error. If `timings` is non-null it receives a per-stage
    // wall-clock breakdown of this call.
    auto Convert(const unsigned char *srcBuffer, StageTimings *timings = nullptr) -> const PlanarRgbFp16 *;

private:
    YuvToRgbConverter() = default;

    auto Deinterleave(const unsigned char *srcBuffer) -> void;

    zimg_filter_graph *_graph = nullptr;

    Kind _kind = Kind::NV12;
    int _width = 0;
    int _height = 0;
    int _subsampleW = 0; // log2 horizontal chroma subsampling
    int _subsampleH = 0; // log2 vertical chroma subsampling
    int _planeBytes = 1; // bytes per planar sample fed to zimg (1 for 8-bit, 2 for 10/16-bit)
    // Aligned planar input buffers fed to zimg: plane 0 = Y, 1 = U, 2 = V.
    void *_p0 = nullptr;
    void *_p1 = nullptr;
    void *_p2 = nullptr;
    ptrdiff_t _stride0 = 0; // bytes
    ptrdiff_t _strideC = 0; // bytes (planes 1 and 2)

    // Pinned planar RGB fp16 output from zimg (padded stride); uploaded to the device directly.
    unsigned short *_r = nullptr;
    unsigned short *_g = nullptr;
    unsigned short *_b = nullptr;
    ptrdiff_t _strideRgb = 0; // bytes

    void *_tmp = nullptr; // zimg scratch buffer

    PlanarRgbFp16 _result; // pointers/stride into _r/_g/_b, returned to the caller
};

}
