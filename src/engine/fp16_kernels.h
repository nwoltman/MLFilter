// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

#include "color/yuv420_format.h"

namespace MLFilter {

// Converts a tightly packed NV12/P010 frame already resident on the device directly to the
// engine's tight planar fp16 NCHW input. Chroma reconstruction matches zimg's Catmull-Rom
// (B=0,C=0.5), mirrored edges, and left/vertically-centred MPEG-2 sample siting.
auto LaunchYuv420ToFp16Planar(const void *dYuv, void *dPlanarFp16, int width, int height,
                              const Yuv420Conversion &conversion, cudaStream_t stream) -> cudaError_t;

// CUDA kernels for the inference pipeline's fp16 tensor I/O: clamping the engine's fp16 input to
// [0,1] before inference, and packing its fp16 planar-RGB output into RGB48 afterward. Each is
// enqueued on the caller's stream so it overlaps the surrounding upload / inference / copy-back.

// Converts the engine's fp16 planar-RGB output (three contiguous R,G,B planes of width*height
// __half each) into packed, top-down RGB48 (uint16 R,G,B per pixel) in a second device buffer.
// Per sample: __half2float -> clamp to [0,1] -> *65535 + 0.5 -> uint16. The destination buffer is
// tightly packed (stride = width*3 uint16); any allocator row padding is applied later by the
// device->host cudaMemcpy2DAsync.
//
// Enqueued on `stream` (no internal synchronization) so it overlaps the surrounding work. Returns
// the result of cudaGetLastError() after the launch.
auto LaunchFp16PlanarToRgb48(const void *dPlanarFp16, void *dPackedRgb48, int width, int height,
                             cudaStream_t stream) -> cudaError_t;

// Clamps `count` fp16 values in `dData` to [0,1] in place, operating directly on the raw 16-bit
// patterns (sign bit set -> 0; > 0x3C00 (1.0) -> 1.0). This is the [0,1] clamp the model input
// needs; applying it on the device lets the YUV->RGB stage upload its planar fp16 unclamped and
// skip a full host pass over the frame.
//
// Enqueued on `stream` (no internal synchronization). Returns cudaGetLastError() after the launch.
auto LaunchClampHalf01(void *dData, size_t count, cudaStream_t stream) -> cudaError_t;

}
