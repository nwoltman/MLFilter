// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

#include "color/yuv420_format.h"

namespace MLFilter {

// Converts a tightly packed NV12/P010 frame already resident on the device directly to the
// engine's tight planar fp16 NCHW input. Chroma reconstruction matches zimg's Catmull-Rom
// (B=0,C=0.5), mirrored edges, and left/vertically-centred MPEG-2 sample siting. Converted RGB
// values are clamped to [0,1] as the final operation before the fp16 store. The clamping is done
// because the YUV->RGB matrix can produce values outside of [0,1] and some models are not trained
// to handle values outside of that range.
auto LaunchYuv420ToFp16Planar(const void *dYuv, void *dPlanarFp16, int width, int height,
                              const Yuv420Conversion &conversion, cudaStream_t stream) -> cudaError_t;

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

}
