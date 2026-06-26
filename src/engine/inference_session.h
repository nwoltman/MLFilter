// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>

namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}

struct CUstream_st;

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

    // Sizes in bytes of the planar-RGB fp16 input/output buffers (3 * W * H * sizeof(half)).
    auto InputBytes() const -> size_t { return static_cast<size_t>(3) * _inW * _inH * sizeof(unsigned short); }
    auto OutputBytes() const -> size_t { return static_cast<size_t>(3) * _outW * _outH * sizeof(unsigned short); }

    // host -> device input, run the network, device output -> host. host buffers are planar
    // RGB fp16 of InputBytes()/OutputBytes(). Returns false on a CUDA/TensorRT error.
    auto Upload(const void *hostInput) -> bool;
    auto Infer() -> bool;
    auto Download(void *hostOutput) -> bool;

private:
    InferenceSession() = default;

    nvinfer1::IRuntime *_runtime = nullptr;
    nvinfer1::ICudaEngine *_engine = nullptr;
    nvinfer1::IExecutionContext *_context = nullptr;
    CUstream_st *_stream = nullptr;

    void *_dInput = nullptr;
    void *_dOutput = nullptr;

    std::string _inputName;
    std::string _outputName;

    int _inW = 0;
    int _inH = 0;
    int _outW = 0;
    int _outH = 0;
};

}
