// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "color/yuv420_format.h"

namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}

struct CUstream_st;
struct CUevent_st;

namespace MLFilter {

// Loads a serialized TensorRT engine and runs inference on the GPU. The engine is the
// static, fp16, NCHW (1x3xHxW) engine produced by TensorRTEngineBuilder; input and output
// are planar RGB fp16 (the input tensor doubles as the engine's NCHW input layout).
//
// Lifetime: one session per pin connection. All CUDA/TensorRT resources are owned and freed
// by the session. Not thread-safe; Transform() is serialized by the filter graph.
class InferenceSession {
public:
    // Deserializes the engine at enginePath and allocates the GPU buffers. Returns nullptr on
    // failure (file missing/unreadable, deserialize failure, or the engine's I/O tensors are
    // not the expected fp16 NCHW C=3 shape); error receives a human-readable reason.
    static auto Create(const std::filesystem::path &enginePath, std::wstring &error) -> std::unique_ptr<InferenceSession>;

    ~InferenceSession();

    InferenceSession(const InferenceSession &) = delete;
    auto operator=(const InferenceSession &) -> InferenceSession & = delete;

    // Resolution of the engine's input (== the connected video resolution) and output (may be
    // larger, e.g. a super-resolution model).
    auto InputWidth() const -> int { return _inW; }
    auto InputHeight() const -> int { return _inH; }
    auto OutputWidth() const -> int { return _outW; }
    auto OutputHeight() const -> int { return _outH; }

    // Sizes in bytes of the planar-RGB fp16 input/output buffers (3 * W * H * sizeof(half)). The
    // packed RGB48 buffer Infer() produces is the same size (also 3 * W * H * 2 bytes).
    auto InputBytes() const -> size_t { return static_cast<size_t>(3) * _inW * _inH * sizeof(unsigned short); }
    auto OutputBytes() const -> size_t { return static_cast<size_t>(3) * _outW * _outH * sizeof(unsigned short); }

    // host -> device input. The three planar RGB fp16 channels (r, g, b) — each
    // InputWidth()*InputHeight() __half with per-row stride srcStrideBytes (may be padded) — are
    // copied into the engine's tight NCHW input buffer and clamped to [0,1] on the device. The
    // source buffers should be pinned for an async DMA. Returns false on a CUDA error.

    // Uploads a compact, tightly packed NV12/P010 frame and performs unpacking, range/depth
    // normalization, Catmull-Rom chroma reconstruction, matrix conversion, [0,1] clamping, and
    // fp16 packing on the GPU.
    auto UploadYuv420(const void *frame, const Yuv420Conversion &conversion) -> bool;

    // Runs the network and, on the same stream, the fp16-planar -> packed-RGB48 conversion kernel.
    // Returns false on a CUDA/TensorRT error.
    auto Infer() -> bool;

    // Copies the packed, top-down RGB48 result from the device into hostOutput, writing each row
    // at dstStrideBytes (the renderer allocator's row pitch, which may exceed width*6). Blocks
    // until the whole stream (upload, inference, conversion, copy) has completed. Returns false on
    // a CUDA error.
    auto Download(void *hostOutput, size_t dstStrideBytes) -> bool;

    // GPU-timeline durations (milliseconds) of the last completed Upload/Infer/Download cycle,
    // measured with CUDA events recorded on the stream. Valid only after a Download() has
    // synchronized the stream. The five intervals are H2D upload, YUV preprocessing, TensorRT
    // inference, RGB48 packing, and D2H download. Returns false on a CUDA error.
    struct GpuStageTimings {
        double uploadMs = 0;
        double preprocessMs = 0;
        double inferenceMs = 0;
        double packMs = 0;
        double downloadMs = 0;
    };
    auto LastGpuTimings(GpuStageTimings &timings) const -> bool;

private:
    InferenceSession() = default;

    nvinfer1::IRuntime *_runtime = nullptr;
    nvinfer1::ICudaEngine *_engine = nullptr;
    nvinfer1::IExecutionContext *_context = nullptr;
    CUstream_st *_stream = nullptr;

    // Stream markers bracketing the five GPU phases reported by LastGpuTimings().
    CUevent_st *_evStart = nullptr;
    CUevent_st *_evUploaded = nullptr;
    CUevent_st *_evPreprocessed = nullptr;
    CUevent_st *_evInferred = nullptr;
    CUevent_st *_evPacked = nullptr;
    CUevent_st *_evDownloaded = nullptr;

    void *_dInput = nullptr;
    void *_dYuv = nullptr;
    void *_dOutput = nullptr;
    void *_dRgb48 = nullptr;

    std::string _inputName;
    std::string _outputName;

    int _inW = 0;
    int _inH = 0;
    int _outW = 0;
    int _outH = 0;
};

}
