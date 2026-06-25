// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "engine_builder.h"
#include "gpu_info.h"

namespace MLFilter {

// %LOCALAPPDATA%\MLFilter\Engines (created if missing). Throws std::filesystem
// errors only on truly unexpected failures; returns empty path if LOCALAPPDATA
// cannot be resolved.
auto GetEnginesDirectory() -> std::filesystem::path;

// Full target path for a static (fp16) engine, encoding model + resolution + GPU +
// driver + TensorRT version so a rebuild after a driver/GPU/TRT change produces a
// distinct file (and the old one can be pruned). Lives in GetEnginesDirectory().
// Engines are always fp16 (the builder converts the model), so precision is not
// encoded in the name.
auto BuildEnginePath(const std::filesystem::path &onnxPath, int width, int height, const GpuInfo &gpu, int trtVersion) -> std::filesystem::path;

// Deletes engines in the Engines dir for this model (across all resolutions) that were
// built for a different GPU/driver/TRT version than `gpu`/`trtVersion` — i.e. anything
// not matching the current environment is stale once it changes. Engines at other
// resolutions for the current environment are retained as valid caches.
auto PruneStaleEngines(const std::filesystem::path &onnxPath, const GpuInfo &gpu, int trtVersion, const ProgressFn &progress) -> void;

}
