// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <functional>
#include <string>

namespace MLFilter {

// A request to build one static engine for a single input resolution.
struct EngineBuildRequest {
    std::filesystem::path onnxPath;
    int width;
    int height;
};

struct EngineBuildResult {
    bool success = false;
    std::filesystem::path enginePath;
    std::wstring message;
};

// Progress/log sink. Implementations call this with human-readable lines.
using ProgressFn = std::function<void(const std::wstring &)>;

// Backend-agnostic engine builder. TensorRT implements it today; a future
// DirectML backend slots in behind the same interface.
class IEngineBuilder {
public:
    virtual ~IEngineBuilder() = default;

    // Path where this request's engine is (or would be) cached, encoding the current
    // GPU/driver/backend-version identity. Empty if the cache dir can't be resolved.
    virtual auto EnginePath(const EngineBuildRequest &request) -> std::filesystem::path = 0;

    // True if a cached engine for this exact request (model/resolution and the current
    // GPU/driver/backend version) already exists on disk.
    virtual auto Exists(const EngineBuildRequest &request) -> bool = 0;

    virtual auto Build(const EngineBuildRequest &request, const ProgressFn &progress) -> EngineBuildResult = 0;
};

}
