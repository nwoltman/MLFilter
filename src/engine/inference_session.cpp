// SPDX-License-Identifier: Apache-2.0

#include "inference_session.h"

#include <cstdint>
#include <fstream>
#include <vector>

#include <cuda_runtime.h>
#include <NvInfer.h>

#include "fp16_kernels.h"

namespace MLFilter {

namespace {

// Minimal logger for the TensorRT runtime. Kept at function-local-static lifetime so it
// outlives every IRuntime that references it. Errors are swallowed (no progress sink at
// runtime); deserialize failures surface through Create()'s error string instead.
class RuntimeLogger : public nvinfer1::ILogger {
public:
    void log(Severity /*severity*/, const char * /*msg*/) noexcept override {}
};

auto Logger() -> RuntimeLogger & {
    static RuntimeLogger logger;
    return logger;
}

// Reads the input/output tensor name, shape and dtype for a single-input/single-output engine.
// Returns false (with error set) if the engine doesn't match the expected fp16 NCHW C=3 shape.
auto ResolveTensor(nvinfer1::ICudaEngine &engine, nvinfer1::TensorIOMode wanted, std::string &name, int &w, int &h, std::wstring &error) -> bool {
    for (int i = 0; i < engine.getNbIOTensors(); ++i) {
        const char *tensorName = engine.getIOTensorName(i);
        if (engine.getTensorIOMode(tensorName) != wanted) {
            continue;
        }

        if (engine.getTensorDataType(tensorName) != nvinfer1::DataType::kHALF) {
            error = L"Engine tensor is not fp16; the model must be fp16.";
            return false;
        }

        const nvinfer1::Dims dims = engine.getTensorShape(tensorName);
        if (dims.nbDims != 4 || dims.d[1] != 3 || dims.d[2] <= 0 || dims.d[3] <= 0) {
            error = L"Engine tensor is not a concrete NCHW (1x3xHxW) shape.";
            return false;
        }

        name = tensorName;
        w = static_cast<int>(dims.d[3]);
        h = static_cast<int>(dims.d[2]);
        return true;
    }

    error = wanted == nvinfer1::TensorIOMode::kINPUT ? L"Engine has no input tensor." : L"Engine has no output tensor.";
    return false;
}

}

auto InferenceSession::Create(const std::filesystem::path &enginePath, std::wstring &error) -> std::unique_ptr<InferenceSession> {
    std::vector<char> engineBytes;
    {
        std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
        if (!file) {
            error = L"Could not open the engine file.";
            return nullptr;
        }
        const std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        engineBytes.resize(static_cast<size_t>(size));
        if (size <= 0 || !file.read(engineBytes.data(), size)) {
            error = L"Could not read the engine file.";
            return nullptr;
        }
    }

    auto session = std::unique_ptr<InferenceSession>(new InferenceSession());

    session->_runtime = nvinfer1::createInferRuntime(Logger());
    if (session->_runtime == nullptr) {
        error = L"Failed to create the TensorRT runtime.";
        return nullptr;
    }

    session->_engine = session->_runtime->deserializeCudaEngine(engineBytes.data(), engineBytes.size());
    if (session->_engine == nullptr) {
        error = L"Failed to deserialize the engine (it may be from an incompatible GPU/driver/TensorRT).";
        return nullptr;
    }

    if (!ResolveTensor(*session->_engine, nvinfer1::TensorIOMode::kINPUT, session->_inputName, session->_inW, session->_inH, error) ||
        !ResolveTensor(*session->_engine, nvinfer1::TensorIOMode::kOUTPUT, session->_outputName, session->_outW, session->_outH, error)) {
        return nullptr;
    }

    session->_context = session->_engine->createExecutionContext();
    if (session->_context == nullptr) {
        error = L"Failed to create the execution context.";
        return nullptr;
    }

    // Pin the (static) input shape so enqueueV3 has fully-specified dimensions.
    if (!session->_context->setInputShape(session->_inputName.c_str(), session->_engine->getTensorShape(session->_inputName.c_str()))) {
        error = L"Failed to set the engine input shape.";
        return nullptr;
    }

    if (cudaMalloc(&session->_dInput, session->InputBytes()) != cudaSuccess ||
        cudaMalloc(&session->_dOutput, session->OutputBytes()) != cudaSuccess ||
        cudaMalloc(&session->_dRgb48, session->OutputBytes()) != cudaSuccess) {
        error = L"Failed to allocate GPU memory for inference.";
        return nullptr;
    }

    if (!session->_context->setTensorAddress(session->_inputName.c_str(), session->_dInput) ||
        !session->_context->setTensorAddress(session->_outputName.c_str(), session->_dOutput)) {
        error = L"Failed to bind the engine I/O buffers.";
        return nullptr;
    }

    if (cudaStreamCreate(&session->_stream) != cudaSuccess) {
        error = L"Failed to create a CUDA stream.";
        return nullptr;
    }

    if (cudaEventCreate(&session->_evStart) != cudaSuccess ||
        cudaEventCreate(&session->_evUploaded) != cudaSuccess ||
        cudaEventCreate(&session->_evInferred) != cudaSuccess ||
        cudaEventCreate(&session->_evDownloaded) != cudaSuccess) {
        error = L"Failed to create CUDA timing events.";
        return nullptr;
    }

    return session;
}

InferenceSession::~InferenceSession() {
    if (_evStart != nullptr) {
        cudaEventDestroy(_evStart);
    }
    if (_evUploaded != nullptr) {
        cudaEventDestroy(_evUploaded);
    }
    if (_evInferred != nullptr) {
        cudaEventDestroy(_evInferred);
    }
    if (_evDownloaded != nullptr) {
        cudaEventDestroy(_evDownloaded);
    }
    if (_stream != nullptr) {
        cudaStreamDestroy(_stream);
    }
    cudaFree(_dInput);
    cudaFree(_dOutput);
    cudaFree(_dRgb48);
    // TensorRT 10/11 objects are released with delete (destroy() was removed).
    delete _context;
    delete _engine;
    delete _runtime;
}

auto InferenceSession::Upload(const void *r, const void *g, const void *b, size_t srcStrideBytes) -> bool {
    // Mark the start of the upload phase on the stream (for the LastGpuTimings() diagnostic).
    cudaEventRecord(_evStart, _stream);

    // Copy each padded-stride planar channel into its (tightly packed) slot of the NCHW input
    // buffer; cudaMemcpy2DAsync drops the source row padding. From pinned memory these run as
    // async DMAs that overlap the rest of the stream.
    const size_t rowBytes = static_cast<size_t>(_inW) * sizeof(unsigned short);
    const size_t planeBytes = rowBytes * _inH;
    auto *dst = static_cast<unsigned char *>(_dInput);
    const void *planes[3] = { r, g, b };
    for (int p = 0; p < 3; ++p) {
        if (cudaMemcpy2DAsync(dst + static_cast<size_t>(p) * planeBytes, rowBytes, planes[p], srcStrideBytes,
                              rowBytes, _inH, cudaMemcpyHostToDevice, _stream) != cudaSuccess) {
            return false;
        }
    }
    // Apply the [0,1] model-input clamp on the device, so the planar fp16 can be uploaded unclamped.
    if (LaunchClampHalf01(_dInput, static_cast<size_t>(3) * _inW * _inH, _stream) != cudaSuccess) {
        return false;
    }
    cudaEventRecord(_evUploaded, _stream);
    return true;
}

auto InferenceSession::Infer() -> bool {
    if (!_context->enqueueV3(_stream)) {
        return false;
    }
    // Convert fp16 planar -> packed RGB48 on the same stream, right after inference, so it overlaps
    // and needs no extra synchronization.
    if (LaunchFp16PlanarToRgb48(_dOutput, _dRgb48, _outW, _outH, _stream) != cudaSuccess) {
        return false;
    }
    cudaEventRecord(_evInferred, _stream);
    return true;
}

auto InferenceSession::Download(void *hostOutput, size_t dstStrideBytes) -> bool {
    // The packed RGB48 result is tightly stored (width*3 uint16 per row); copy it straight into the
    // output sample, expanding each row to the allocator's (possibly padded) destination pitch.
    const size_t rowBytes = static_cast<size_t>(_outW) * 3 * sizeof(uint16_t);
    if (cudaMemcpy2DAsync(hostOutput, dstStrideBytes, _dRgb48, rowBytes, rowBytes, _outH,
                          cudaMemcpyDeviceToHost, _stream) != cudaSuccess) {
        return false;
    }
    cudaEventRecord(_evDownloaded, _stream);
    return cudaStreamSynchronize(_stream) == cudaSuccess;
}

auto InferenceSession::LastGpuTimings(GpuStageTimings &timings) const -> bool {
    // The stream is synchronized by Download(), so all four markers have completed. Each interval is
    // the GPU-timeline duration of that phase: upload (H2D + clamp), compute (enqueueV3 + RGB48
    // kernel), download (D2H copy).
    float upload = 0, compute = 0, download = 0;
    if (cudaEventElapsedTime(&upload, _evStart, _evUploaded) != cudaSuccess ||
        cudaEventElapsedTime(&compute, _evUploaded, _evInferred) != cudaSuccess ||
        cudaEventElapsedTime(&download, _evInferred, _evDownloaded) != cudaSuccess) {
        return false;
    }
    timings.uploadMs = upload;
    timings.computeMs = compute;
    timings.downloadMs = download;
    return true;
}

}
