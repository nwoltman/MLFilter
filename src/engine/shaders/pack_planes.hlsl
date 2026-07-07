// SPDX-License-Identifier: Apache-2.0
// Packs a native D3D11 YUV 4:2:0 texture into a CUDA-readable byte buffer.

#ifndef SAMPLE_BYTES
#error SAMPLE_BYTES must be defined as 1 for NV12 or 2 for P010
#endif

#if SAMPLE_BYTES == 1
#define SAMPLE_SCALE 255.0f
#elif SAMPLE_BYTES == 2
#define SAMPLE_SCALE 65535.0f
#else
#error SAMPLE_BYTES must be 1 for NV12 or 2 for P010
#endif

cbuffer CopyConstants : register(b0) {
    uint width;
    uint height;
    uint rowPitch;
    uint yPlaneBytes;
};

Texture2DArray<float> srcY : register(t0);
Texture2DArray<float2> srcUv : register(t1);
RWByteAddressBuffer dst : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint samplesPerWord = 4 / SAMPLE_BYTES;
    const uint wordsPerRow = (width * SAMPLE_BYTES + 3) / 4;
    if (id.x >= wordsPerRow || id.y >= height) {
        return;
    }

    const uint firstSample = id.x * samplesPerWord;
    const uint sampleMask = SAMPLE_BYTES == 1 ? 0xff : 0xffff;

    uint packed = 0;
    [unroll]
    for (uint yLane = 0; yLane < samplesPerWord; ++yLane) {
        const uint x = firstSample + yLane;
        if (x >= width) {
            break;
        }

        const float value = srcY.Load(int4(x, id.y, 0, 0));
        const uint code = (uint)round(saturate(value) * SAMPLE_SCALE) & sampleMask;
        packed |= code << (yLane * SAMPLE_BYTES * 8);
    }
    dst.Store(id.y * rowPitch + id.x * 4, packed);

    if (id.y >= height / 2) {
        return;
    }

    packed = 0;
    [unroll]
    for (uint pairLane = 0; pairLane < samplesPerWord / 2; ++pairLane) {
        const uint x = firstSample + pairLane * 2;
        if (x >= width) {
            break;
        }

        const float2 pair = srcUv.Load(int4(x / 2, id.y, 0, 0));
        const uint2 code = (uint2)round(saturate(pair) * SAMPLE_SCALE) & sampleMask;
        const uint firstLane = pairLane * 2;
        packed |= code.x << (firstLane * SAMPLE_BYTES * 8);
        packed |= code.y << ((firstLane + 1) * SAMPLE_BYTES * 8);
    }
    dst.Store(yPlaneBytes + id.y * rowPitch + id.x * 4, packed);
}
