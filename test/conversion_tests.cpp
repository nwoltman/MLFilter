// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "color/yuv420_format.h"
#include "engine/fp16_kernels.h"
#include "helpers/zimg_reference.h"

namespace {

constexpr int kWidth = 1920;
constexpr int kHeight = 1080;

enum class Pattern {
    Gradient,
    Random,
    Extremes,
};

auto PatternName(Pattern pattern) -> const char * {
    switch (pattern) {
    case Pattern::Gradient: return "gradient";
    case Pattern::Random: return "random";
    case Pattern::Extremes: return "extremes";
    }
    return "unknown";
}

struct DeviceBuffer {
    void *data = nullptr;
    ~DeviceBuffer() { cudaFree(data); }
};

auto Hash(uint32_t value) -> uint32_t {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

auto ScaleCoordinate(size_t coordinate, size_t extent, int minimum, int maximum) -> int {
    if (extent <= 1) return (minimum + maximum) / 2;
    return minimum + static_cast<int>(
        (static_cast<uint64_t>(coordinate) * (maximum - minimum) + (extent - 1) / 2) /
        (extent - 1));
}

template <typename Sample>
auto MakeLimitedFrame(int depth, Pattern pattern) -> std::vector<unsigned char> {
    const int shift = depth == 10 ? 6 : 0;
    const int yMinimum = 16 << (depth - 8);
    const int yMaximum = 235 << (depth - 8);
    const int cMinimum = 16 << (depth - 8);
    const int cMaximum = 240 << (depth - 8);
    const size_t sampleCount = static_cast<size_t>(kWidth) * kHeight * 3 / 2;
    std::vector<unsigned char> frame(sampleCount * sizeof(Sample));
    auto *samples = reinterpret_cast<Sample *>(frame.data());

    const size_t lumaCount = static_cast<size_t>(kWidth) * kHeight;
    for (size_t i = 0; i < lumaCount; ++i) {
        const int x = static_cast<int>(i % kWidth);
        const int y = static_cast<int>(i / kWidth);
        int code;
        if (pattern == Pattern::Gradient) {
            code = ScaleCoordinate(i, lumaCount, yMinimum, yMaximum);
        } else if (pattern == Pattern::Random) {
            code = yMinimum + static_cast<int>(Hash(static_cast<uint32_t>(i)) %
                                               (yMaximum - yMinimum + 1));
        } else {
            // Alternating legal extrema plus narrow stripes exercise clamping after matrix
            // conversion and prevent adjacent chroma blocks from sharing similar luma.
            code = ((x ^ y) & 1) ? yMaximum : yMinimum;
            if ((x % 31) == 0) code = (yMinimum + yMaximum) / 2;
        }
        samples[i] = static_cast<Sample>(code << shift);
    }

    Sample *uv = samples + lumaCount;
    const int chromaWidth = kWidth / 2;
    const int chromaHeight = kHeight / 2;
    for (int y = 0; y < chromaHeight; ++y) {
        for (int x = 0; x < chromaWidth; ++x) {
            const size_t pair = static_cast<size_t>(y) * chromaWidth + x;
            int cb;
            int cr;
            if (pattern == Pattern::Gradient) {
                cb = ScaleCoordinate(x, chromaWidth, cMinimum, cMaximum);
                cr = ScaleCoordinate(y, chromaHeight, cMinimum, cMaximum);
            } else if (pattern == Pattern::Random) {
                cb = cMinimum + static_cast<int>(
                    Hash(static_cast<uint32_t>(pair * 2 + lumaCount)) %
                    (cMaximum - cMinimum + 1));
                cr = cMinimum + static_cast<int>(
                    Hash(static_cast<uint32_t>(pair * 2 + 1 + lumaCount)) %
                    (cMaximum - cMinimum + 1));
            } else {
                // Opposing checkerboards maximize Catmull-Rom overshoot and exercise both
                // horizontal and vertical quarter phases, including mirrored frame edges.
                cb = ((x + y) & 1) ? cMaximum : cMinimum;
                cr = ((x - y) & 1) ? cMinimum : cMaximum;
                if (x == 0 || x == chromaWidth - 1 ||
                    y == 0 || y == chromaHeight - 1) {
                    cb = (x & 1) ? cMaximum : cMinimum;
                    cr = (y & 1) ? cMaximum : cMinimum;
                }
            }
            uv[pair * 2] = static_cast<Sample>(cb << shift);
            uv[pair * 2 + 1] = static_cast<Sample>(cr << shift);
        }
    }
    return frame;
}

auto ClampHalf(uint16_t value) -> uint16_t {
    if (value & 0x8000u) return 0;
    if (value > 0x3c00u) return 0x3c00u;
    return value;
}

auto RunConversionTest(MLFilter::Yuv420Format format, bool bt709, Pattern pattern) -> bool {
    const bool p010 = format == MLFilter::Yuv420Format::P010;
    std::vector<unsigned char> frame =
        p010 ? MakeLimitedFrame<uint16_t>(10, pattern)
             : MakeLimitedFrame<uint8_t>(8, pattern);

    MLFilter::YuvToRgbConverter::Params referenceParams {
        .kind = p010 ? MLFilter::YuvToRgbConverter::Kind::P010
                     : MLFilter::YuvToRgbConverter::Kind::NV12,
        .width = kWidth,
        .height = kHeight,
        .bt709 = bt709,
        .fullRange = false,
    };
    std::wstring error;
    auto referenceConverter = MLFilter::YuvToRgbConverter::Create(referenceParams, error);
    if (!referenceConverter) {
        wprintf(L"zimg setup failed: %ls\n", error.c_str());
        return false;
    }
    const auto *reference = referenceConverter->Convert(frame.data());
    if (!reference) {
        printf("zimg conversion failed\n");
        return false;
    }

    DeviceBuffer deviceInput;
    DeviceBuffer deviceOutput;
    const size_t outputBytes = static_cast<size_t>(3) * kWidth * kHeight * sizeof(uint16_t);
    if (cudaMalloc(&deviceInput.data, frame.size()) != cudaSuccess ||
        cudaMalloc(&deviceOutput.data, outputBytes) != cudaSuccess ||
        cudaMemcpy(deviceInput.data, frame.data(), frame.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("CUDA allocation/upload failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return false;
    }

    const MLFilter::Yuv420Conversion conversion {
        .format = format,
        .bt709 = bt709,
        .fullRange = false,
    };
    const size_t outputSamples = static_cast<size_t>(3) * kWidth * kHeight;
    if (MLFilter::LaunchYuv420ToFp16Planar(
            deviceInput.data, deviceOutput.data, kWidth, kHeight, conversion, nullptr) != cudaSuccess) {
        printf("CUDA conversion launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return false;
    }

    std::vector<uint16_t> gpu(outputSamples);
    if (cudaMemcpy(gpu.data(), deviceOutput.data, outputBytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("CUDA result download failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return false;
    }

    const uint16_t *planes[3] = {
        static_cast<const uint16_t *>(reference->r),
        static_cast<const uint16_t *>(reference->g),
        static_cast<const uint16_t *>(reference->b),
    };
    size_t exact = 0;
    unsigned maximumDelta = 0;
    int maximumPlane = 0, maximumX = 0, maximumY = 0;
    for (int plane = 0; plane < 3; ++plane) {
        for (int y = 0; y < kHeight; ++y) {
            const auto *row = reinterpret_cast<const uint16_t *>(
                reinterpret_cast<const unsigned char *>(planes[plane]) +
                static_cast<ptrdiff_t>(y) * reference->strideBytes);
            for (int x = 0; x < kWidth; ++x) {
                const uint16_t expected = ClampHalf(row[x]);
                const uint16_t actual =
                    gpu[(static_cast<size_t>(plane) * kHeight + y) * kWidth + x];
                if (expected == actual) ++exact;
                const unsigned delta = expected > actual ? expected - actual : actual - expected;
                if (delta > maximumDelta) {
                    maximumDelta = delta;
                    maximumPlane = plane;
                    maximumX = x;
                    maximumY = y;
                }
            }
        }
    }

    const char *formatName = p010 ? "P010" : "NV12";
    const char *matrixName = bt709 ? "BT.709" : "BT.601";
    printf("  %s %s %-8s: %.5f%% exact, max FP16 code delta %u\n",
           formatName, matrixName, PatternName(pattern),
           100.0 * exact / outputSamples, maximumDelta);
    if (maximumDelta > 1) {
        printf("    mismatch at plane %d, x=%d, y=%d\n",
               maximumPlane, maximumX, maximumY);
        return false;
    }
    return true;
}

}

auto main() -> int {
    printf("MLFilter GPU conversion tests\n");
    int failures = 0;
    int total = 0;
    const auto run = [&](MLFilter::Yuv420Format format, bool bt709, Pattern pattern) {
        ++total;
        if (!RunConversionTest(format, bt709, pattern)) ++failures;
    };
    for (const Pattern pattern : { Pattern::Gradient, Pattern::Random, Pattern::Extremes }) {
        run(MLFilter::Yuv420Format::NV12, false, pattern);
        run(MLFilter::Yuv420Format::NV12, true, pattern);
        run(MLFilter::Yuv420Format::P010, false, pattern);
        run(MLFilter::Yuv420Format::P010, true, pattern);
    }
    printf("\n%d passed, %d failed\n", total - failures, failures);
    return failures == 0 ? 0 : 1;
}
