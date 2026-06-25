// SPDX-License-Identifier: Apache-2.0

#include "tensorrt_engine_builder.h"

#include <fstream>
#include <format>
#include <memory>
#include <vector>

#include <windows.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "engine_naming.h"
#include "gpu_info.h"
#include "onnx_fp16.h"

namespace MLFilter {

namespace {

auto ToUtf8(const std::wstring &str) -> std::string {
    if (str.empty()) {
        return {};
    }
    const int needed = WideCharToMultiByte(CP_UTF8, 0, str.data(), static_cast<int>(str.size()), nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(needed), '\0');
    WideCharToMultiByte(CP_UTF8, 0, str.data(), static_cast<int>(str.size()), out.data(), needed, nullptr, nullptr);
    return out;
}

// Routes TensorRT's internal log messages (errors + warnings) to the progress sink.
class ProgressLogger : public nvinfer1::ILogger {
public:
    explicit ProgressLogger(const ProgressFn &progress)
        : _progress(progress) {}

    void log(Severity severity, const char *msg) noexcept override {
        if (severity > Severity::kWARNING || !_progress || msg == nullptr) {
            return;
        }
        const wchar_t *prefix = severity == Severity::kWARNING ? L"[TRT warning] " : L"[TRT error] ";
        const int needed = MultiByteToWideChar(CP_UTF8, 0, msg, -1, nullptr, 0);
        if (needed <= 0) {
            return;
        }
        std::wstring wide(static_cast<size_t>(needed - 1), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, msg, -1, wide.data(), needed);
        _progress(prefix + wide);
    }

private:
    const ProgressFn &_progress;
};

template <typename T>
using TrtPtr = std::unique_ptr<T>;

auto Fail(std::wstring message) -> EngineBuildResult {
    return EngineBuildResult { .success = false, .enginePath = {}, .message = std::move(message) };
}

}

auto TensorRTEngineBuilder::EnginePath(const EngineBuildRequest &request) -> std::filesystem::path {
    return BuildEnginePath(request.onnxPath, request.width, request.height, QueryGpuInfo(), getInferLibVersion());
}

auto TensorRTEngineBuilder::Exists(const EngineBuildRequest &request) -> bool {
    const std::filesystem::path enginePath = EnginePath(request);
    return !enginePath.empty() && std::filesystem::exists(enginePath);
}

auto TensorRTEngineBuilder::Build(const EngineBuildRequest &request, const ProgressFn &progress) -> EngineBuildResult {
    const auto log = [&](const std::wstring &message) {
        if (progress) {
            progress(message);
        }
    };

    if (!std::filesystem::exists(request.onnxPath)) {
        return Fail(std::format(L"ONNX model not found: {}", request.onnxPath.wstring()));
    }

    log(std::format(L"Building {}x{} engine from {}", request.width, request.height, request.onnxPath.filename().wstring()));

    ProgressLogger logger(progress);

    TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) {
        return Fail(L"Failed to create TensorRT builder.");
    }

    TrtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    if (!network) {
        return Fail(L"Failed to create TensorRT network.");
    }

    TrtPtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser) {
        return Fail(L"Failed to create ONNX parser.");
    }

    // Read the ONNX model and convert it to fp16. TensorRT 11 networks are strongly
    // typed, so the engine precision follows the model's types; converting up front is
    // what yields an fp16 engine from an fp32 model.
    std::vector<uint8_t> modelBytes;
    {
        std::ifstream modelFile(request.onnxPath, std::ios::binary | std::ios::ate);
        if (!modelFile) {
            return Fail(std::format(L"Could not open ONNX model: {}", request.onnxPath.wstring()));
        }
        const std::streamsize size = modelFile.tellg();
        modelFile.seekg(0, std::ios::beg);
        modelBytes.resize(static_cast<size_t>(size));
        if (size > 0 && !modelFile.read(reinterpret_cast<char *>(modelBytes.data()), size)) {
            return Fail(L"Failed to read the ONNX model file.");
        }
    }

    const OnnxFp16Result fp16 = ConvertOnnxToFp16(modelBytes);
    if (fp16.changed) {
        log(L"Converted model to fp16.");
    } else if (!fp16.ok) {
        log(fp16.message);
    }

    // model_path lets the parser resolve any external weight files relative to the model.
    const std::string onnxPathUtf8 = ToUtf8(request.onnxPath.wstring());
    if (!parser->parse(fp16.bytes.data(), fp16.bytes.size(), onnxPathUtf8.c_str())) {
        return Fail(L"Failed to parse ONNX model. See log for details.");
    }

    if (network->getNbInputs() < 1) {
        return Fail(L"ONNX model has no inputs.");
    }

    TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) {
        return Fail(L"Failed to create builder config.");
    }

    config->setBuilderOptimizationLevel(5);

    // TensorRT 11 networks are always strongly typed: the engine precision follows
    // the ONNX model's types and cannot be forced via a builder flag. We detect the
    // input precision and reflect it in the engine filename.
    nvinfer1::ITensor *input = network->getInput(0);
    const nvinfer1::Dims inputDims = input->getDimensions();
    if (inputDims.nbDims != 4) {
        return Fail(std::format(L"Expected a 4D NCHW model input, got {} dimensions.", inputDims.nbDims));
    }

    // Static engine for the selected resolution: fill any dynamic (-1) dimensions
    // from the requested NCHW shape; keep dimensions the model already fixes.
    nvinfer1::Dims dims = inputDims;
    if (dims.d[0] < 0) {
        dims.d[0] = 1;
    }
    if (dims.d[1] < 0) {
        dims.d[1] = 3;
    }
    if (dims.d[2] < 0) {
        dims.d[2] = request.height;
    }
    if (dims.d[3] < 0) {
        dims.d[3] = request.width;
    }

    // Single optimization profile with min == opt == max -> a static engine.
    nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();
    const char *inputName = input->getName();
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, dims);
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, dims);
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, dims);
    config->addOptimizationProfile(profile);

    if (input->getType() != nvinfer1::DataType::kHALF) {
        log(L"Warning: model input is still not fp16 after conversion; the engine may not be fp16.");
    }

    log(L"Optimizing (this can take a while)...");

    TrtPtr<nvinfer1::IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
    if (!serialized || serialized->size() == 0) {
        return Fail(L"TensorRT failed to build the engine.");
    }

    const GpuInfo gpu = QueryGpuInfo();
    const std::filesystem::path enginePath = BuildEnginePath(request.onnxPath, request.width, request.height, gpu, getInferLibVersion());
    if (enginePath.empty()) {
        return Fail(L"Could not resolve the engine output directory (%LOCALAPPDATA%).");
    }

    // Remove engines from a prior driver/GPU/TRT version for this model
    PruneStaleEngines(request.onnxPath, gpu, getInferLibVersion(), progress);

    std::ofstream file(enginePath, std::ios::binary | std::ios::trunc);
    if (!file) {
        return Fail(std::format(L"Could not open engine file for writing: {}", enginePath.wstring()));
    }
    file.write(static_cast<const char *>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
    file.close();
    if (!file) {
        return Fail(std::format(L"Failed while writing engine file: {}", enginePath.wstring()));
    }

    log(std::format(L"Built: {}", enginePath.filename().wstring()));

    return EngineBuildResult { .success = true, .enginePath = enginePath, .message = L"OK" };
}

}
