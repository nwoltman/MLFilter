// SPDX-License-Identifier: Apache-2.0

#include "fp16_to_rgb48.h"

#include <cstdint>

#include <cuda_fp16.h>

namespace MLFilter {

namespace {

// fp16 [0,1]-ish -> 16-bit full-range integer. The engine output should sit in [0,1] but clamp
// defensively before scaling. Rounds to nearest, ties to even (IEEE-754 default, via the
// __float2uint_rn intrinsic) so the result is bit-identical to zimg's float->uint16 quantization
// and free of the upward tie bias / double-rounding wart of the old (f*65535 + 0.5) approach.
__device__ inline auto HalfToUnorm16(__half h) -> uint16_t {
    float f = __half2float(h);
    f = fminf(fmaxf(f, 0.0f), 1.0f);
    return static_cast<uint16_t>(__float2uint_rn(f * 65535.0f));
}

// One thread per output pixel. Reads the three planar fp16 channels and writes one interleaved
// R,G,B uint16 triple. Top-down packing (row y at dst + y*width*3), matching the RGB48 'RGB0'
// convention and the channel order LAV/avisynth_filter use.
__global__ void Fp16PlanarToRgb48Kernel(const __half *r, const __half *g, const __half *b,
                                        uint16_t *dst, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const size_t i = static_cast<size_t>(y) * width + x;
    uint16_t *px = dst + i * 3;
    px[0] = HalfToUnorm16(r[i]);
    px[1] = HalfToUnorm16(g[i]);
    px[2] = HalfToUnorm16(b[i]);
}

}

auto LaunchFp16PlanarToRgb48(const void *dPlanarFp16, void *dPackedRgb48, int width, int height,
                             cudaStream_t stream) -> cudaError_t {
    const auto *r = static_cast<const __half *>(dPlanarFp16);
    const __half *g = r + static_cast<size_t>(width) * height;
    const __half *b = g + static_cast<size_t>(width) * height;

    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    Fp16PlanarToRgb48Kernel<<<grid, block, 0, stream>>>(r, g, b, static_cast<uint16_t *>(dPackedRgb48),
                                                        width, height);
    return cudaGetLastError();
}

}
