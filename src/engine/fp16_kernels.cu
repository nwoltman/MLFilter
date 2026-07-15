// SPDX-License-Identifier: Apache-2.0

#include "fp16_kernels.h"

#include <cstdint>

#include <cuda_fp16.h>

namespace MLFilter {

namespace {

constexpr float chromaScale8BitFull = 1.0f / 255.0f;
constexpr float chromaScale10BitFull = 1.0f / 1023.0f;
constexpr float chromaScale8BitLimited = 1.0f / 224.0f;
constexpr float chromaScale10BitLimited = 1.0f / 896.0f;
constexpr float chromaOffset8BitFull = -128.0f / 255.0f;
constexpr float chromaOffset10BitFull = -512.0f / 1023.0f;
constexpr float chromaOffset8BitLimited = -128.0f / 224.0f;
constexpr float chromaOffset10BitLimited = -512.0f / 896.0f;

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

constexpr int kChromaBlockWidth = 32;
constexpr int kChromaBlockHeight = 8;
constexpr int kChromaTileWidth = kChromaBlockWidth + 3;
constexpr int kChromaTileHeight = kChromaBlockHeight + 4;

__device__ __forceinline__ auto Clamp01(float value) -> float {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

__device__ inline auto MirrorIndex(int i, int length) -> int {
    if (i >= 0 && i < length) {
        return i;
    }

    if (length <= 1) {
        return 0;
    }

    const int period = 2 * length;
    i %= period;
    if (i < 0) {
        i += period;
    }

    return i < length ? i : period - i - 1;
}

struct ChromaQuad {
    float2 topLeft;
    float2 topRight;
    float2 bottomLeft;
    float2 bottomRight;
};

template <bool BOTTOM>
__device__ __forceinline__ auto FilterVerticalChroma(const float2 rows[5]) -> float2 {
    constexpr int start = BOTTOM ? 1 : 0;
    constexpr float w0 = BOTTOM ? -0.0703125f : -0.0234375f;
    constexpr float w1 = BOTTOM ?  0.8671875f :  0.2265625f;
    constexpr float w2 = BOTTOM ?  0.2265625f :  0.8671875f;
    constexpr float w3 = BOTTOM ? -0.0234375f : -0.0703125f;

    float2 even {
        w0 * rows[start].x,
        w0 * rows[start].y,
    };
    even.x = fmaf(w2, rows[start + 2].x, even.x);
    even.y = fmaf(w2, rows[start + 2].y, even.y);
    float2 odd {
        w1 * rows[start + 1].x,
        w1 * rows[start + 1].y,
    };
    odd.x = fmaf(w3, rows[start + 3].x, odd.x);
    odd.y = fmaf(w3, rows[start + 3].y, odd.y);
    return make_float2(even.x + odd.x, even.y + odd.y);
}

template <typename TUV, int SHIFT>
__device__ __forceinline__ auto StageBufferChromaTile(
        const unsigned char *uv,
        size_t uvPitchBytes,
        int chromaWidth,
        int chromaHeight,
        float scale,
        float offset,
        float2 tile[kChromaTileHeight][kChromaTileWidth]) -> void {
    const int threadIndex = threadIdx.y * blockDim.x + threadIdx.x;
    const int threadCount = blockDim.x * blockDim.y;
    constexpr int tileSamples = kChromaTileWidth * kChromaTileHeight;
    const int blockCx = blockIdx.x * kChromaBlockWidth;
    const int blockCy = blockIdx.y * kChromaBlockHeight;

    for (int index = threadIndex; index < tileSamples; index += threadCount) {
        const int tileY = index / kChromaTileWidth;
        const int tileX = index - tileY * kChromaTileWidth;
        const int sx = MirrorIndex(blockCx + tileX - 1, chromaWidth);
        const int sy = MirrorIndex(blockCy + tileY - 2, chromaHeight);
        const auto *row = reinterpret_cast<const TUV *>(
            uv + static_cast<size_t>(sy) * uvPitchBytes);
        const TUV code = row[sx];

        tile[tileY][tileX] = make_float2(
            fmaf(static_cast<float>(code.x >> SHIFT), scale, offset),
            fmaf(static_cast<float>(code.y >> SHIFT), scale, offset));
    }
}

__device__ __forceinline__ auto ReconstructSharedChromaQuad(
        const float2 tile[kChromaTileHeight][kChromaTileWidth]) -> ChromaQuad {
    // A thread's aligned 2x2 luma block needs four samples from each of five chroma rows.
    // Those samples are already normalized and shared by every thread in this CUDA block.
    float2 leftRows[5];
    float2 rightRows[5];

#pragma unroll
    for (int j = 0; j < 5; ++j) {
        const int tileY = threadIdx.y + j;
        const int tileX = threadIdx.x;
        const float2 first = tile[tileY][tileX];
        const float2 second = tile[tileY][tileX + 1];
        const float2 third = tile[tileY][tileX + 2];
        const float2 fourth = tile[tileY][tileX + 3];

        leftRows[j] = second;

        // Match zimg's AVX2 reduction tree: even and odd taps are accumulated
        // independently with FMA, then added before the vertical reconstruction.
        float2 even {
            -0.0625f * first.x,
            -0.0625f * first.y,
        };
        even.x = fmaf(0.5625f, third.x, even.x);
        even.y = fmaf(0.5625f, third.y, even.y);
        float2 odd {
            0.5625f * second.x,
            0.5625f * second.y,
        };
        odd.x = fmaf(-0.0625f, fourth.x, odd.x);
        odd.y = fmaf(-0.0625f, fourth.y, odd.y);
        rightRows[j] = make_float2(even.x + odd.x, even.y + odd.y);
    }

    return {
        FilterVerticalChroma<false>(leftRows),
        FilterVerticalChroma<false>(rightRows),
        FilterVerticalChroma<true>(leftRows),
        FilterVerticalChroma<true>(rightRows),
    };
}

__device__ __forceinline__ auto ConvertYuvToRgb(float yy,
                                                float2 chroma,
                                                float rCr,
                                                float gCb,
                                                float gCr,
                                                float bCb) -> float3 {
    // Clamp final values to the [0,1] range to accommodate models that aren't trained to handle
    // RGB values outside of that range
    return make_float3(
        Clamp01(fmaf(rCr, chroma.y, yy)),
        Clamp01(fmaf(gCr, chroma.y, fmaf(gCb, chroma.x, yy))),
        Clamp01(fmaf(bCb, chroma.x, yy)));
}

__device__ __forceinline__ auto StoreRgbPair(__half *dst,
                                             size_t plane,
                                             size_t i,
                                             float3 left,
                                             float3 right,
                                             bool hasRight) -> void {
    // The base allocation is aligned. All three plane addresses are half2-aligned when both
    // the plane size and the first sample index are even.
    if (hasRight && ((plane | i) & 1) == 0) {
        *reinterpret_cast<__half2 *>(dst + i) = __floats2half2_rn(left.x, right.x);
        *reinterpret_cast<__half2 *>(dst + plane + i) = __floats2half2_rn(left.y, right.y);
        *reinterpret_cast<__half2 *>(dst + 2 * plane + i) = __floats2half2_rn(left.z, right.z);
        return;
    }

    dst[i] = __float2half_rn(left.x);
    dst[plane + i] = __float2half_rn(left.y);
    dst[2 * plane + i] = __float2half_rn(left.z);

    if (hasRight) {
        dst[i + 1] = __float2half_rn(right.x);
        dst[plane + i + 1] = __float2half_rn(right.y);
        dst[2 * plane + i + 1] = __float2half_rn(right.z);
    }
}

template <typename TY, typename TUV, int SHIFT>
__global__ void Yuv420ToFp16PlanarKernel(const unsigned char *src,
                                         size_t yPitchBytes,
                                         size_t uvOffsetBytes,
                                         size_t uvPitchBytes,
                                         __half *dst,
                                         int width,
                                         int height,
                                         bool bt709,
                                         bool fullRange) {
    // Each thread owns one aligned 2x2 luma block so its four outputs share chroma work.
    const int x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int y = 2 * (blockIdx.y * blockDim.y + threadIdx.y);
    const unsigned char *uv = src + uvOffsetBytes;

    float chromaScale;
    float chromaOffset;
    if constexpr (sizeof(TY) == 2) {
        chromaScale = fullRange ? chromaScale10BitFull : chromaScale10BitLimited;
        chromaOffset = fullRange ? chromaOffset10BitFull : chromaOffset10BitLimited;
    } else {
        chromaScale = fullRange ? chromaScale8BitFull : chromaScale8BitLimited;
        chromaOffset = fullRange ? chromaOffset8BitFull : chromaOffset8BitLimited;
    }

    __shared__ float2 chromaTile[kChromaTileHeight][kChromaTileWidth];
    StageBufferChromaTile<TUV, SHIFT>(
        uv, uvPitchBytes, width / 2, height / 2,
        chromaScale, chromaOffset, chromaTile);
    __syncthreads();

    if (x >= width || y >= height) {
        return;
    }

    const int depth = sizeof(TY) == 1 ? 8 : 10;
    const float codeScale = static_cast<float>(1 << (depth - 8));
    const float maximum = static_cast<float>((1 << depth) - 1);
    const float yOffset = fullRange ? 0.0f : 16.0f * codeScale;
    const float yRange = fullRange ? maximum : 219.0f * codeScale;
    // zimg stores scale/offset as float and evaluates scale*code+offset with FMA.
    const float yScale = 1.0f / yRange;
    const float yBias = -yOffset / yRange;
    const auto sampleY = [&](int sx, int sy) {
        const auto *row = reinterpret_cast<const TY *>(
            src + static_cast<size_t>(sy) * yPitchBytes);
        return fmaf(static_cast<float>(row[sx] >> SHIFT), yScale, yBias);
    };

    const ChromaQuad chroma = ReconstructSharedChromaQuad(chromaTile);

    // Coefficients are the float casts of zimg's double-precision BT.709/BT.470BG inverse
    // matrices. Preserve zimg's multiply, FMA, FMA evaluation order.
    const float rCr = bt709 ? kBt709RedFromCr : kBt601RedFromCr;
    const float gCb = bt709 ? kBt709GreenFromCb : kBt601GreenFromCb;
    const float gCr = bt709 ? kBt709GreenFromCr : kBt601GreenFromCr;
    const float bCb = bt709 ? kBt709BlueFromCb : kBt601BlueFromCb;

    const bool hasRight = x + 1 < width;
    const bool hasBottom = y + 1 < height;
    const size_t plane = static_cast<size_t>(width) * height;
    const size_t top = static_cast<size_t>(y) * width + x;

    const float3 topLeft = ConvertYuvToRgb(
        sampleY(x, y), chroma.topLeft, rCr, gCb, gCr, bCb);
    const float3 topRight = hasRight
        ? ConvertYuvToRgb(sampleY(x + 1, y), chroma.topRight, rCr, gCb, gCr, bCb)
        : topLeft;
    StoreRgbPair(dst, plane, top, topLeft, topRight, hasRight);

    if (hasBottom) {
        const float3 bottomLeft = ConvertYuvToRgb(
            sampleY(x, y + 1), chroma.bottomLeft, rCr, gCb, gCr, bCb);
        const float3 bottomRight = hasRight
            ? ConvertYuvToRgb(
                sampleY(x + 1, y + 1), chroma.bottomRight, rCr, gCb, gCr, bCb)
            : bottomLeft;
        StoreRgbPair(dst, plane, top + width, bottomLeft, bottomRight, hasRight);
    }
}

// fp16 [0,1]-ish -> 16-bit full-range integer. The engine output should sit in [0,1] but clamp
// defensively before scaling. Rounds to nearest, ties to even (IEEE-754 default, via the
// __float2uint_rn intrinsic) so the result is bit-identical to zimg's float->uint16 quantization.
__device__ __forceinline__ auto HalfToUnorm16(__half h) -> uint16_t {
    const float f = __saturatef(__half2float(h));
    return static_cast<uint16_t>(__float2uint_rn(f * 65535.0f));
}

constexpr unsigned kRgb48BlockSize = 256;
constexpr unsigned kRgb48ComponentsPerPixel = 3;
constexpr unsigned kRgb48ComponentsPerVector = sizeof(uint4) / sizeof(uint16_t);
constexpr unsigned kRgb48TileComponents = kRgb48BlockSize * kRgb48ComponentsPerPixel;

// One thread per output pixel converts the three planar channels into an interleaved shared-memory
// tile. The block then writes that tile as contiguous uint4 vectors. Linear top-down packing matches
// the RGB48 'RGB0' convention.
__global__ void Fp16PlanarToRgb48Kernel(const __half *__restrict__ r,
                                        const __half *__restrict__ g,
                                        const __half *__restrict__ b,
                                        uint16_t *__restrict__ dst,
                                        size_t pixelCount) {
    __shared__ __align__(16) uint16_t packed[kRgb48TileComponents];

    const unsigned thread = threadIdx.x;
    const size_t blockPixel = static_cast<size_t>(blockIdx.x) * kRgb48BlockSize;
    const size_t i = blockPixel + thread;
    if (i < pixelCount) {
        packed[thread * kRgb48ComponentsPerPixel] = HalfToUnorm16(r[i]);
        packed[thread * kRgb48ComponentsPerPixel + 1] = HalfToUnorm16(g[i]);
        packed[thread * kRgb48ComponentsPerPixel + 2] = HalfToUnorm16(b[i]);
    }
    __syncthreads();

    const size_t remainingPixels = pixelCount - blockPixel;
    const unsigned blockPixels = remainingPixels < kRgb48BlockSize
        ? static_cast<unsigned>(remainingPixels)
        : kRgb48BlockSize;
    const unsigned componentCount = blockPixels * kRgb48ComponentsPerPixel;
    const unsigned vectorCount = componentCount / kRgb48ComponentsPerVector;
    const size_t blockComponent = blockPixel * kRgb48ComponentsPerPixel;

    if (thread < vectorCount) {
        auto *dstVectors = reinterpret_cast<uint4 *>(dst + blockComponent);
        const auto *packedVectors = reinterpret_cast<const uint4 *>(packed);

        dstVectors[thread] = packedVectors[thread];
    }

    const unsigned vectorComponents = vectorCount * kRgb48ComponentsPerVector;
    const unsigned tailComponents = componentCount - vectorComponents;
    if (thread < tailComponents) {
        dst[blockComponent + vectorComponents + thread] = packed[vectorComponents + thread];
    }
}

}

auto LaunchYuv420ToFp16Planar(const void *dYuv,
                              size_t yPitchBytes,
                              size_t uvOffsetBytes,
                              size_t uvPitchBytes,
                              void *dPlanarFp16,
                              int width,
                              int height,
                              const Yuv420Conversion &conversion,
                              cudaStream_t stream) -> cudaError_t {
    const dim3 block(kChromaBlockWidth, kChromaBlockHeight);
    const dim3 grid((width + 2 * block.x - 1) / (2 * block.x),
                    (height + 2 * block.y - 1) / (2 * block.y));

    if (conversion.format == Yuv420Format::NV12) {
        Yuv420ToFp16PlanarKernel<unsigned char, uchar2, 0><<<grid, block, 0, stream>>>(
            static_cast<const unsigned char *>(dYuv),
            yPitchBytes, uvOffsetBytes, uvPitchBytes,
            static_cast<__half *>(dPlanarFp16),
            width, height, conversion.bt709, conversion.fullRange);
    } else {
        Yuv420ToFp16PlanarKernel<unsigned short, ushort2, 6><<<grid, block, 0, stream>>>(
            static_cast<const unsigned char *>(dYuv),
            yPitchBytes, uvOffsetBytes, uvPitchBytes,
            static_cast<__half *>(dPlanarFp16),
            width, height, conversion.bt709, conversion.fullRange);
    }

    return cudaGetLastError();
}

auto LaunchFp16PlanarToRgb48(const void *dPlanarFp16, void *dPackedRgb48, int width, int height,
                             cudaStream_t stream) -> cudaError_t {
    const size_t pixelCount = static_cast<size_t>(width) * height;
    const auto *r = static_cast<const __half *>(dPlanarFp16);
    const __half *g = r + pixelCount;
    const __half *b = g + pixelCount;

    const dim3 block(kRgb48BlockSize);
    const dim3 grid(static_cast<unsigned>((pixelCount + block.x - 1) / block.x));
    Fp16PlanarToRgb48Kernel<<<grid, block, 0, stream>>>(r, g, b, static_cast<uint16_t *>(dPackedRgb48),
                                                        pixelCount);
    return cudaGetLastError();
}

}
