// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "engine_builder.h"

namespace MLFilter {

// Builds static, fp16, single-resolution TensorRT engines from an ONNX model
// using the native TensorRT C++ API (nvinfer + nvonnxparser).
class TensorRTEngineBuilder : public IEngineBuilder {
public:
    auto EnginePath(const EngineBuildRequest &request) -> std::filesystem::path override;
    auto Exists(const EngineBuildRequest &request) -> bool override;
    auto Build(const EngineBuildRequest &request, const ProgressFn &progress) -> EngineBuildResult override;
};

}
