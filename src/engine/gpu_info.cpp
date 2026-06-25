// SPDX-License-Identifier: Apache-2.0

#include "gpu_info.h"

#include <array>
#include <cstdio>
#include <memory>

#include <windows.h>

#include <nvml.h>

namespace MLFilter {

namespace {

auto ToWide(const char *str) -> std::wstring {
    if (str == nullptr || *str == '\0') {
        return {};
    }

    const int needed = MultiByteToWideChar(CP_UTF8, 0, str, -1, nullptr, 0);
    if (needed <= 0) {
        return {};
    }

    std::wstring out(static_cast<size_t>(needed - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, str, -1, out.data(), needed);
    return out;
}

auto Trim(std::wstring s) -> std::wstring {
    const auto isWs = [](wchar_t c) { return c == L' ' || c == L'\t' || c == L'\r' || c == L'\n'; };
    while (!s.empty() && isWs(s.front())) {
        s.erase(s.begin());
    }
    while (!s.empty() && isWs(s.back())) {
        s.pop_back();
    }
    return s;
}

// Primary path: NVML (nvml.dll ships with the NVIDIA driver).
auto QueryViaNvml() -> GpuInfo {
    GpuInfo info;

    if (nvmlInit_v2() != NVML_SUCCESS) {
        return info;
    }

    nvmlDevice_t device {};
    if (nvmlDeviceGetHandleByIndex_v2(0, &device) == NVML_SUCCESS) {
        std::array<char, NVML_DEVICE_NAME_BUFFER_SIZE> name {};
        if (nvmlDeviceGetName(device, name.data(), static_cast<unsigned int>(name.size())) == NVML_SUCCESS) {
            info.name = ToWide(name.data());
        }
    }

    std::array<char, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE> driver {};
    if (nvmlSystemGetDriverVersion(driver.data(), static_cast<unsigned int>(driver.size())) == NVML_SUCCESS) {
        info.driverVersion = ToWide(driver.data());
    }

    nvmlShutdown();

    info.available = !info.name.empty() && !info.driverVersion.empty();
    return info;
}

// Fallback: parse `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader`.
auto QueryViaNvidiaSmi() -> GpuInfo {
    GpuInfo info;

    std::unique_ptr<FILE, decltype(&_pclose)> pipe(
        _wpopen(L"nvidia-smi --query-gpu=name,driver_version --format=csv,noheader", L"rt"), &_pclose);
    if (!pipe) {
        return info;
    }

    std::wstring out;
    std::array<wchar_t, 512> buffer {};
    while (fgetws(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
        out += buffer.data();
    }

    // Take the first line (device 0), split on the first comma.
    if (const size_t newline = out.find_first_of(L"\r\n"); newline != std::wstring::npos) {
        out.resize(newline);
    }
    if (const size_t comma = out.find(L','); comma != std::wstring::npos) {
        info.name = Trim(out.substr(0, comma));
        info.driverVersion = Trim(out.substr(comma + 1));
        info.available = !info.name.empty() && !info.driverVersion.empty();
    }

    return info;
}

}

auto QueryGpuInfo() -> GpuInfo {
    if (GpuInfo info = QueryViaNvml(); info.available) {
        return info;
    }
    return QueryViaNvidiaSmi();
}

}
