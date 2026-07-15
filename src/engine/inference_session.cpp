// SPDX-License-Identifier: Apache-2.0

#include "inference_session.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <NvInfer.h>

#include "d3d11_cuda_input.h"
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
        cudaMalloc(&session->_dYuv, static_cast<size_t>(session->_inW) * session->_inH * 3) != cudaSuccess ||
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

#ifdef MLFILTER_ENABLE_STAGE_TIMINGS
    if (cudaEventCreate(&session->_evStart) != cudaSuccess ||
        cudaEventCreate(&session->_evUploaded) != cudaSuccess ||
        cudaEventCreate(&session->_evPreprocessed) != cudaSuccess ||
        cudaEventCreate(&session->_evInferred) != cudaSuccess ||
        cudaEventCreate(&session->_evOutput) != cudaSuccess) {
        error = L"Failed to create CUDA timing events.";
        return nullptr;
    }
#endif

    return session;
}

InferenceSession::~InferenceSession() {
    UnregisterOutputBuffers();
    _d3d11Input.reset();

#ifdef MLFILTER_ENABLE_STAGE_TIMINGS
    if (_evStart != nullptr) {
        cudaEventDestroy(_evStart);
    }
    if (_evUploaded != nullptr) {
        cudaEventDestroy(_evUploaded);
    }
    if (_evPreprocessed != nullptr) {
        cudaEventDestroy(_evPreprocessed);
    }
    if (_evInferred != nullptr) {
        cudaEventDestroy(_evInferred);
    }
    if (_evOutput != nullptr) {
        cudaEventDestroy(_evOutput);
    }
#endif
    if (_stream != nullptr) {
        cudaStreamDestroy(_stream);
    }
    cudaFree(_dInput);
    cudaFree(_dYuv);
    cudaFree(_dOutput);
    cudaFree(_dRgb48);
    // TensorRT 10/11 objects are released with delete (destroy() was removed).
    delete _context;
    delete _engine;
    delete _runtime;
}

auto InferenceSession::UploadYuv420(const void *frame, const Yuv420Conversion &conversion) -> bool {
#ifdef MLFILTER_ENABLE_STAGE_TIMINGS
    cudaEventRecord(_evStart, _stream);
#endif

    const size_t sampleBytes = conversion.format == Yuv420Format::P010 ? 2 : 1;
    const size_t pitchBytes = static_cast<size_t>(_inW) * sampleBytes;
    const size_t yBytes = pitchBytes * _inH;
    const size_t frameBytes = yBytes + pitchBytes * (_inH / 2);
    if (cudaMemcpyAsync(_dYuv, frame, frameBytes, cudaMemcpyHostToDevice, _stream) != cudaSuccess) {
        return false;
    }

#ifdef MLFILTER_ENABLE_STAGE_TIMINGS
    cudaEventRecord(_evUploaded, _stream);
#endif

    if (LaunchYuv420ToFp16Planar(
            _dYuv, pitchBytes, yBytes, pitchBytes, _dInput,
            _inW, _inH, conversion, _stream) != cudaSuccess) {
        return false;
    }

#ifdef MLFILTER_ENABLE_STAGE_TIMINGS
    cudaEventRecord(_evPreprocessed, _stream);
#endif

    return true;
}

auto InferenceSession::UploadD3D11Yuv420(ID3D11Texture2D *texture,
                                         unsigned arraySlice,
                                         ID3D11Device *device,
                                         ID3D11DeviceContext *context,
                                         HANDLE contextMutex,
                                         const Yuv420Conversion &conversion) -> bool {
#ifdef MLFILTER_ENABLE_STAGE_TIMINGS
    cudaEventRecord(_evStart, _stream);
#endif

    if (_d3d11Input == nullptr) {
        _d3d11Input = std::make_unique<D3D11CudaInput>(_inW, _inH);
    }

    return _d3d11Input->Upload(texture, arraySlice, device, context, contextMutex, conversion,
                               _dInput, _stream
#ifdef MLFILTER_ENABLE_STAGE_TIMINGS
                               , _evUploaded, _evPreprocessed
#endif
    );
}

auto InferenceSession::Infer() -> bool {
    if (!_context->enqueueV3(_stream)) {
        return false;
    }
#ifdef MLFILTER_ENABLE_STAGE_TIMINGS
    cudaEventRecord(_evInferred, _stream);
#endif
    return true;
}

auto InferenceSession::AcquireMappedOutput(void *hostOutput,
                                           size_t hostBufferBytes,
                                           size_t requiredBytes) -> MappedOutput {
    if (hostBufferBytes < requiredBytes) {
        return {};
    }

    const auto cached = std::find_if(
        _hostRegistrations.begin(), _hostRegistrations.end(),
        [hostOutput](const HostRegistration &item) { return item.address == hostOutput; });
    if (cached != _hostRegistrations.end()) {
        return cached->bytes >= requiredBytes
            ? MappedOutput {.deviceAddress = cached->deviceAddress}
            : MappedOutput {};
    }

    if (cudaHostRegister(hostOutput, hostBufferBytes, cudaHostRegisterMapped) != cudaSuccess) {
        return {};
    }

    void *deviceOutput = nullptr;
    if (cudaHostGetDevicePointer(&deviceOutput, hostOutput, 0) != cudaSuccess) {
        cudaHostUnregister(hostOutput);
        return {};
    }

    if (_hostRegistrations.size() < kMaxCachedRegistrations) {
        _hostRegistrations.push_back({hostOutput, deviceOutput, hostBufferBytes});
        return {.deviceAddress = deviceOutput};
    }

    ++_outputTransientTransfers;
    return {.deviceAddress = deviceOutput, .transient = true};
}

auto InferenceSession::QueueMappedOutput(void *deviceOutput, size_t dstStrideBytes) -> bool {
    return LaunchFp16PlanarToRgb48(
        _dOutput, deviceOutput, dstStrideBytes, _outW, _outH, _stream) == cudaSuccess;
}

auto InferenceSession::QueuePageableOutput(void *hostOutput,
                                           size_t dstStrideBytes,
                                           size_t rowBytes) -> bool {
    if (LaunchFp16PlanarToRgb48(
            _dOutput, _dRgb48, rowBytes, _outW, _outH, _stream) != cudaSuccess) {
        return false;
    }

    return cudaMemcpy2DAsync(hostOutput, dstStrideBytes, _dRgb48, rowBytes, rowBytes, _outH,
                             cudaMemcpyDeviceToHost, _stream) == cudaSuccess;
}

auto InferenceSession::Download(void *hostOutput, size_t dstStrideBytes, size_t hostBufferBytes) -> bool {
    const size_t rowBytes = static_cast<size_t>(_outW) * 3 * sizeof(uint16_t);
    if (hostOutput == nullptr ||
        (reinterpret_cast<uintptr_t>(hostOutput) & (alignof(uint16_t) - 1)) != 0 ||
        dstStrideBytes < rowBytes ||
        (dstStrideBytes & (alignof(uint16_t) - 1)) != 0) {
        return false;
    }

    const size_t requiredBytes = dstStrideBytes * static_cast<size_t>(_outH);
    const MappedOutput mapped =
        AcquireMappedOutput(hostOutput, hostBufferBytes, requiredBytes);
    const bool outputQueued = mapped.deviceAddress != nullptr
        ? QueueMappedOutput(mapped.deviceAddress, dstStrideBytes)
        : QueuePageableOutput(hostOutput, dstStrideBytes, rowBytes);

    bool completed = false;
    if (outputQueued) {
#ifdef MLFILTER_ENABLE_STAGE_TIMINGS
        cudaEventRecord(_evOutput, _stream);
#endif
        completed = cudaStreamSynchronize(_stream) == cudaSuccess;
    }

    if (mapped.transient) {
        cudaHostUnregister(hostOutput);
    }

    return completed;
}

auto InferenceSession::UnregisterOutputBuffers() -> void {
    for (const HostRegistration &registration : _hostRegistrations) {
        cudaHostUnregister(registration.address);
    }

    _hostRegistrations.clear();
    _outputTransientTransfers = 0;
}

auto InferenceSession::GetOutputCacheStatus() const -> OutputCacheStatus {
    return {
        .cached = _hostRegistrations.size(),
        .capacity = kMaxCachedRegistrations,
        .transientTransfers = _outputTransientTransfers,
    };
}

#ifdef MLFILTER_ENABLE_STAGE_TIMINGS
auto InferenceSession::LastGpuTimings(GpuStageTimings &timings) const -> bool {
    // Download() synchronized the stream, so all markers have completed. Adjacent event pairs
    // isolate H2D, preprocessing, TensorRT, and fused RGB48 host output on the GPU timeline.
    float upload = 0, preprocess = 0, inference = 0, output = 0;
    if (cudaEventElapsedTime(&upload, _evStart, _evUploaded) != cudaSuccess ||
        cudaEventElapsedTime(&preprocess, _evUploaded, _evPreprocessed) != cudaSuccess ||
        cudaEventElapsedTime(&inference, _evPreprocessed, _evInferred) != cudaSuccess ||
        cudaEventElapsedTime(&output, _evInferred, _evOutput) != cudaSuccess) {
        return false;
    }

    timings.uploadMs = upload;
    timings.preprocessMs = preprocess;
    timings.inferenceMs = inference;
    timings.outputMs = output;

    return true;
}
#endif

}
