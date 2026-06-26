// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

namespace MLFilter {

// Converts the engine's fp16 planar-RGB output (three contiguous R,G,B planes of width*height
// __half each) into packed, top-down RGB48 (uint16 R,G,B per pixel) in a second device buffer.
// Per sample: __half2float -> clamp to [0,1] -> *65535 + 0.5 -> uint16 (the exact arithmetic the
// CPU HalfToUnorm16 path used). The destination buffer is tightly packed (stride = width*3
// uint16); any allocator row padding is applied later by the device->host cudaMemcpy2DAsync.
//
// Enqueued on `stream` (no internal synchronization) so it overlaps the surrounding work. Returns
// the result of cudaGetLastError() after the launch.
auto LaunchFp16PlanarToRgb48(const void *dPlanarFp16, void *dPackedRgb48, int width, int height,
                             cudaStream_t stream) -> cudaError_t;

}
