// SPDX-License-Identifier: Apache-2.0

#include "engine_naming.h"

#include <array>
#include <cwctype>
#include <format>
#include <system_error>

#include <windows.h>

namespace MLFilter {

namespace {

// Replace characters that are awkward in filenames: spaces -> '_', dots -> '-',
// any other non-alphanumeric (keeping '_' and '-') -> '_'.
auto SanitizeToken(std::wstring_view token) -> std::wstring {
    std::wstring out;
    out.reserve(token.size());
    for (const wchar_t c : token) {
        if (std::iswalnum(c)) {
            out += c;
        } else if (c == L'.') {
            out += L'-';
        } else {
            out += L'_';
        }
    }
    return out;
}

// Stable 32-bit FNV-1a over the lowercased absolute path, as 8 hex chars. Keeps
// engines from different models with the same stem from colliding in the shared
// Engines directory.
auto OnnxPathHash(const std::filesystem::path &onnxPath) -> std::wstring {
    std::error_code ec;
    std::filesystem::path absolute = std::filesystem::absolute(onnxPath, ec);
    std::wstring key = (ec ? onnxPath : absolute).wstring();
    for (wchar_t &c : key) {
        c = static_cast<wchar_t>(std::towlower(c));
    }

    uint32_t hash = 2166136261u;
    for (const wchar_t c : key) {
        hash ^= static_cast<uint32_t>(c & 0xffff);
        hash *= 16777619u;
    }

    return std::format(L"{:08x}", hash);
}

// "{stem}.{hash}." — the prefix shared by every engine for this model, across all
// resolutions and GPU/driver/TRT versions.
auto ModelPrefix(const std::filesystem::path &onnxPath) -> std::wstring {
    return std::format(L"{}.{}.", onnxPath.stem().wstring(), OnnxPathHash(onnxPath));
}

// "{stem}.{hash}.{gpu}.drv{driver}.trt{trt}." — the prefix shared by every engine for
// this model built in the current GPU/driver/TRT environment, across all resolutions.
// Engines whose name does not start with this were built for a prior environment and
// are stale once it changes.
auto EnvPrefix(const std::filesystem::path &onnxPath, const GpuInfo &gpu, int trtVersion) -> std::wstring {
    const std::wstring gpuToken = gpu.available ? SanitizeToken(gpu.name) : L"unknownGPU";
    const std::wstring driverToken = gpu.available ? SanitizeToken(gpu.driverVersion) : L"unknownDriver";
    return std::format(L"{}{}.drv{}.trt{}.", ModelPrefix(onnxPath), gpuToken, driverToken, trtVersion);
}

}

auto GetEnginesDirectory() -> std::filesystem::path {
    std::array<wchar_t, MAX_PATH> buffer {};
    const DWORD len = GetEnvironmentVariableW(L"LOCALAPPDATA", buffer.data(), static_cast<DWORD>(buffer.size()));
    if (len == 0 || len >= buffer.size()) {
        return {};
    }

    std::filesystem::path dir = std::filesystem::path(buffer.data()) / L"MLFilter" / L"Engines";

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir;
}

auto BuildEnginePath(const std::filesystem::path &onnxPath, int width, int height, const GpuInfo &gpu, int trtVersion) -> std::filesystem::path {
    const std::wstring fileName = std::format(L"{}{}x{}.engine", EnvPrefix(onnxPath, gpu, trtVersion), width, height);
    return GetEnginesDirectory() / fileName;
}

auto PruneStaleEngines(const std::filesystem::path &onnxPath, const GpuInfo &gpu, int trtVersion, const ProgressFn &progress) -> void {
    const std::filesystem::path dir = GetEnginesDirectory();
    if (dir.empty()) {
        return;
    }

    // An engine for this model (any resolution) whose name doesn't start with the current
    // environment prefix was built for a prior GPU/driver/TRT version and is stale.
    // Current-environment engines at other resolutions share the prefix and are kept --
    // including the just-built one, so it never needs special handling here.
    const std::wstring modelPrefix = ModelPrefix(onnxPath);
    const std::wstring envPrefix = EnvPrefix(onnxPath, gpu, trtVersion);

    std::error_code ec;
    for (const auto &entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file()) {
            continue;
        }

        const std::wstring name = entry.path().filename().wstring();
        const bool matchesModel = name.starts_with(modelPrefix) && name.ends_with(L".engine");
        if (!matchesModel || name.starts_with(envPrefix)) {
            continue;
        }

        std::error_code removeEc;
        if (std::filesystem::remove(entry.path(), removeEc) && progress) {
            progress(std::format(L"Removed stale engine: {}", name));
        }
    }
}

}
