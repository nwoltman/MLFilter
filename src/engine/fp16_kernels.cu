// SPDX-License-Identifier: Apache-2.0

#include "fp16_kernels.h"

#include <cstdint>

#include <cuda_fp16.h>

namespace MLFilter {

namespace {

// Non-constant-luminance Y'CbCr -> R'G'B' matrix coefficients. Luma has coefficient 1.0
// for every output channel; omitted chroma coefficients are zero.
constexpr float kBt709RedFromCr = 1.5748f;
constexpr float kBt709GreenFromCb = -0.1873242729306488f;
constexpr float kBt709GreenFromCr = -0.46812427293064884f;
constexpr float kBt709BlueFromCb = 1.8556f;

constexpr float kBt601RedFromCr = 1.402f;
constexpr float kBt601GreenFromCb = -0.34413628620102216f;
constexpr float kBt601GreenFromCr = -0.7141362862010221f;
constexpr float kBt601BlueFromCb = 1.772f;

__device__ __forceinline__ auto Clamp01(float value) -> float {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

__device__ inline auto MirrorIndex(int i, int length) -> int {
    if (i < 0) {
        return -i - 1;
    }
    if (i >= length) {
        return 2 * length - i - 1;
    }
    return i;
}

template <typename T>
__device__ auto ReconstructFloatChroma(const T *uv, int chromaWidth, int chromaHeight,
                                       int x, int y, int component, int shift,
                                       float scale, float offset) -> float {
    const int cx = x >> 1;
    const int cy = y >> 1;
    // zimg treats MPEG-2 4:2:0 chroma as vertically centred between each pair of
    // luma rows. Consequently, even and odd output rows sample different phases of
    // the Catmull-Rom filter: even rows use the quarter phase and odd rows use the
    // three-quarter phase. The four-tap source window therefore begins one row
    // earlier for even rows. `y & 1` selects both the window and phase coefficients.
    const int y0 = cy - ((y & 1) ? 1 : 2);
    float rows[4];
    // Catmull-Rom bicubic reconstruction (B=0, C=0.5). `vy` contains the two
    // vertical quarter-phase four-tap filters; the horizontal half-phase taps
    // below are [-1/16, 9/16, 9/16, -1/16].
    const float vy[4] = {
        (y & 1) ? -0.0703125f : -0.0234375f,
        (y & 1) ?  0.8671875f :  0.2265625f,
        (y & 1) ?  0.2265625f :  0.8671875f,
        (y & 1) ? -0.0234375f : -0.0703125f,
    };
    const auto sample = [&](int sx, int sy) {
        sx = MirrorIndex(sx, chromaWidth);
        sy = MirrorIndex(sy, chromaHeight);
        const float code = static_cast<float>(
            uv[(static_cast<size_t>(sy) * chromaWidth + sx) * 2 + component] >> shift);
        return fmaf(code, scale, offset);
    };
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if ((x & 1) == 0) {
            rows[j] = sample(cx, y0 + j);
        } else {
            // Match zimg's AVX2 reduction tree: even and odd taps are accumulated
            // independently with FMA, then added before the fp16 store.
            float even = -0.0625f * sample(cx - 1, y0 + j);
            even = fmaf(0.5625f, sample(cx + 1, y0 + j), even);
            float odd = 0.5625f * sample(cx, y0 + j);
            odd = fmaf(-0.0625f, sample(cx + 2, y0 + j), odd);
            rows[j] = even + odd;
        }
    }
    float even = vy[0] * rows[0];
    even = fmaf(vy[2], rows[2], even);
    float odd = vy[1] * rows[1];
    odd = fmaf(vy[3], rows[3], odd);
    return even + odd;
}

template <typename T>
__global__ void Yuv420ToFp16PlanarKernel(const T *src, __half *dst, int width, int height,
                                         int shift, bool bt709, bool fullRange) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const size_t i = static_cast<size_t>(y) * width + x;
    const T *uv = src + static_cast<size_t>(width) * height;
    const int depth = sizeof(T) == 1 ? 8 : 10;
    const float codeScale = static_cast<float>(1 << (depth - 8));
    const float maximum = static_cast<float>((1 << depth) - 1);
    const float yOffset = fullRange ? 0.0f : 16.0f * codeScale;
    const float yRange = fullRange ? maximum : 219.0f * codeScale;
    // zimg stores scale/offset as float and evaluates scale*code+offset with FMA.
    const float yy = fmaf(static_cast<float>(src[i] >> shift), 1.0f / yRange, -yOffset / yRange);
    float cb, cr;
    if constexpr (sizeof(T) == 2) {
        const float scale = fullRange ? 1.0f / 1023.0f : 1.0f / 896.0f;
        const float offset = fullRange ? -512.0f / 1023.0f : -512.0f / 896.0f;
        cb = ReconstructFloatChroma(uv, width / 2, height / 2, x, y, 0, shift,
                                    scale, offset);
        cr = ReconstructFloatChroma(uv, width / 2, height / 2, x, y, 1, shift,
                                    scale, offset);
    } else {
        if (fullRange) {
            cb = ReconstructFloatChroma(uv, width / 2, height / 2, x, y, 0, shift,
                                        1.0f / 255.0f, -128.0f / 255.0f);
            cr = ReconstructFloatChroma(uv, width / 2, height / 2, x, y, 1, shift,
                                        1.0f / 255.0f, -128.0f / 255.0f);
        } else {
            cb = ReconstructFloatChroma(uv, width / 2, height / 2, x, y, 0, shift,
                                        1.0f / 224.0f, -128.0f / 224.0f);
            cr = ReconstructFloatChroma(uv, width / 2, height / 2, x, y, 1, shift,
                                        1.0f / 224.0f, -128.0f / 224.0f);
        }
    }

    // Coefficients are the float casts of zimg's double-precision BT.709/BT.470BG inverse
    // matrices. Preserve zimg's multiply, FMA, FMA evaluation order.
    const float rCr = bt709 ? kBt709RedFromCr : kBt601RedFromCr;
    const float gCb = bt709 ? kBt709GreenFromCb : kBt601GreenFromCb;
    const float gCr = bt709 ? kBt709GreenFromCr : kBt601GreenFromCr;
    const float bCb = bt709 ? kBt709BlueFromCb : kBt601BlueFromCb;

    // Clamp final values to the [0,1] range to accommodate models that aren't trained to handle
    // RGB values outside of that range.
    const float r = Clamp01(fmaf(rCr, cr, yy));
    const float g = Clamp01(fmaf(gCr, cr, fmaf(gCb, cb, yy)));
    const float b = Clamp01(fmaf(bCb, cb, yy));

    const size_t plane = static_cast<size_t>(width) * height;
    dst[i] = __float2half_rn(r);
    dst[plane + i] = __float2half_rn(g);
    dst[2 * plane + i] = __float2half_rn(b);
}

// fp16 [0,1]-ish -> 16-bit full-range integer. The engine output should sit in [0,1] but clamp
// defensively before scaling. Rounds to nearest, ties to even (IEEE-754 default, via the
// __float2uint_rn intrinsic) so the result is bit-identical to zimg's float->uint16 quantization.
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

auto LaunchYuv420ToFp16Planar(const void *dYuv, void *dPlanarFp16, int width, int height,
                              const Yuv420Conversion &conversion, cudaStream_t stream) -> cudaError_t {
    const dim3 block(32, 8);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    if (conversion.format == Yuv420Format::NV12) {
        Yuv420ToFp16PlanarKernel<<<grid, block, 0, stream>>>(
            static_cast<const uint8_t *>(dYuv), static_cast<__half *>(dPlanarFp16),
            width, height, 0, conversion.bt709, conversion.fullRange);
    } else {
        Yuv420ToFp16PlanarKernel<<<grid, block, 0, stream>>>(
            static_cast<const uint16_t *>(dYuv), static_cast<__half *>(dPlanarFp16),
            width, height, 6, conversion.bt709, conversion.fullRange);
    }
    return cudaGetLastError();
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
