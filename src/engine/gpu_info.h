// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace MLFilter {

struct GpuInfo {
    bool available = false;
    std::wstring name;            // e.g. "NVIDIA GeForce RTX 3080"
    std::wstring driverVersion;   // e.g. "596.36"
};

// Queries the primary GPU (device 0) name and driver version via NVML, falling
// back to spawning nvidia-smi. Returns available == false if neither works.
auto QueryGpuInfo() -> GpuInfo;

}
