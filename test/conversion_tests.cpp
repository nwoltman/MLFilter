// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <string>
#include <vector>

#include <cuda_d3d11_interop.h>
#include <cuda_runtime.h>
#include <d3d11.h>
#include <dxgi.h>

#include "color/yuv420_format.h"
#include "engine/d3d11_cuda_input.h"
#include "engine/fp16_kernels.h"
#include "helpers/zimg_reference.h"

namespace {

constexpr int kWidth = 1920;
constexpr int kHeight = 1080;

template <typename T>
class ComRef {
public:
    ComRef() = default;

    ~ComRef() {
        Reset();
    }

    ComRef(const ComRef &) = delete;
    auto operator=(const ComRef &) -> ComRef & = delete;

    auto Address() -> T ** {
        Reset();
        return &_value;
    }

    auto Get() const -> T * {
        return _value;
    }

    auto Reset() -> void {
        if (_value != nullptr) {
            _value->Release();
            _value = nullptr;
        }
    }

private:
    T *_value = nullptr;
};

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

struct HostRegistration {
    void *address = nullptr;
    ~HostRegistration() {
        if (address != nullptr) {
            cudaHostUnregister(address);
        }
    }
};

struct CudaEvent {
    cudaEvent_t value = nullptr;
    ~CudaEvent() {
        if (value != nullptr) {
            cudaEventDestroy(value);
        }
    }
};

auto FindCudaDxgiAdapter(IDXGIAdapter **adapter) -> bool {
    *adapter = nullptr;

    int currentCudaDevice = 0;
    if (cudaGetDevice(&currentCudaDevice) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    ComRef<IDXGIFactory> factory;
    if (FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory),
                                 reinterpret_cast<void **>(factory.Address())))) {
        return false;
    }

    for (UINT index = 0;; ++index) {
        IDXGIAdapter *candidate = nullptr;
        if (factory.Get()->EnumAdapters(index, &candidate) == DXGI_ERROR_NOT_FOUND) {
            break;
        }

        int adapterCudaDevice = -1;
        const cudaError_t result = cudaD3D11GetDevice(&adapterCudaDevice, candidate);
        if (result == cudaSuccess && adapterCudaDevice == currentCudaDevice) {
            *adapter = candidate;
            return true;
        }

        if (result != cudaSuccess) {
            cudaGetLastError();
        }
        candidate->Release();
    }

    return false;
}

auto CreateD3D11TestDevice(ComRef<ID3D11Device> &device,
                           ComRef<ID3D11DeviceContext> &context) -> bool {
    ComRef<IDXGIAdapter> adapter;
    if (!FindCudaDxgiAdapter(adapter.Address())) {
        return false;
    }

    const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_0};
    D3D_FEATURE_LEVEL createdLevel = D3D_FEATURE_LEVEL_11_0;
    return SUCCEEDED(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
                                       D3D11_CREATE_DEVICE_SINGLETHREADED,
                                       levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
                                       device.Address(), &createdLevel, context.Address()));
}

auto CreateD3D11InputTexture(ID3D11Device *device,
                             const std::vector<unsigned char> &frame,
                             MLFilter::Yuv420Format format,
                             ComRef<ID3D11Texture2D> &texture) -> bool {
    const size_t sampleBytes = format == MLFilter::Yuv420Format::P010 ? 2 : 1;

    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = kWidth;
    desc.Height = kHeight;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format == MLFilter::Yuv420Format::P010
        ? DXGI_FORMAT_P010
        : DXGI_FORMAT_NV12;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    D3D11_SUBRESOURCE_DATA initial {};
    initial.pSysMem = frame.data();
    initial.SysMemPitch = static_cast<UINT>(static_cast<size_t>(kWidth) * sampleBytes);
    initial.SysMemSlicePitch = static_cast<UINT>(frame.size());
    return SUCCEEDED(device->CreateTexture2D(&desc, &initial, texture.Address()));
}

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

struct ConversionComparison {
    size_t exact = 0;
    unsigned maximumDelta = 0;
    int maximumPlane = 0;
    int maximumX = 0;
    int maximumY = 0;
};

auto CompareWithReference(const std::vector<uint16_t> &actual,
                          const uint16_t *const planes[3],
                          ptrdiff_t strideBytes) -> ConversionComparison {
    ConversionComparison comparison;

    for (int plane = 0; plane < 3; ++plane) {
        for (int y = 0; y < kHeight; ++y) {
            const auto *row = reinterpret_cast<const uint16_t *>(
                reinterpret_cast<const unsigned char *>(planes[plane]) +
                static_cast<ptrdiff_t>(y) * strideBytes);
            for (int x = 0; x < kWidth; ++x) {
                const uint16_t expected = ClampHalf(row[x]);
                const uint16_t value =
                    actual[(static_cast<size_t>(plane) * kHeight + y) * kWidth + x];
                if (expected == value) {
                    ++comparison.exact;
                }
                const unsigned delta = expected > value ? expected - value : value - expected;
                if (delta > comparison.maximumDelta) {
                    comparison.maximumDelta = delta;
                    comparison.maximumPlane = plane;
                    comparison.maximumX = x;
                    comparison.maximumY = y;
                }
            }
        }
    }

    return comparison;
}

auto RunConversionTest(MLFilter::Yuv420Format format,
                       bool bt709,
                       Pattern pattern,
                       ID3D11Device *d3dDevice,
                       ID3D11DeviceContext *d3dContext,
                       MLFilter::D3D11CudaInput &d3dInput,
                       cudaEvent_t uploadedEvent,
                       cudaEvent_t preprocessedEvent) -> bool {
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

    DeviceBuffer deviceTightInput;
    DeviceBuffer deviceOutput;
    const size_t outputBytes = static_cast<size_t>(3) * kWidth * kHeight * sizeof(uint16_t);
    const size_t sampleBytes = p010 ? 2 : 1;
    const size_t yBytes = static_cast<size_t>(kWidth) * kHeight * sampleBytes;
    const size_t rowBytes = static_cast<size_t>(kWidth) * sampleBytes;

    if (cudaMalloc(&deviceTightInput.data, frame.size()) != cudaSuccess ||
        cudaMalloc(&deviceOutput.data, outputBytes) != cudaSuccess) {
        printf("CUDA allocation/upload failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return false;
    }

    if (cudaMemcpy(deviceTightInput.data, frame.data(), frame.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("CUDA upload failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return false;
    }

    const MLFilter::Yuv420Conversion conversion {
        .format = format,
        .bt709 = bt709,
        .fullRange = false,
    };
    const size_t outputSamples = static_cast<size_t>(3) * kWidth * kHeight;
    if (MLFilter::LaunchYuv420ToFp16Planar(
            deviceTightInput.data, rowBytes, yBytes, rowBytes,
            deviceOutput.data, kWidth, kHeight, conversion, nullptr) != cudaSuccess) {
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
    const ConversionComparison softwareComparison =
        CompareWithReference(gpu, planes, reference->strideBytes);

    const char *formatName = p010 ? "P010" : "NV12";
    const char *matrixName = bt709 ? "BT.709" : "BT.601";
    printf("  %s %s %-8s %-8s: %.5f%% exact, max FP16 code delta %u\n",
           formatName, matrixName, PatternName(pattern), "software",
           100.0 * softwareComparison.exact / outputSamples,
           softwareComparison.maximumDelta);
    if (softwareComparison.maximumDelta > 1) {
        printf("    mismatch at plane %d, x=%d, y=%d\n",
               softwareComparison.maximumPlane,
               softwareComparison.maximumX,
               softwareComparison.maximumY);
        return false;
    }

    ComRef<ID3D11Texture2D> d3dTexture;
    if (!CreateD3D11InputTexture(d3dDevice, frame, format, d3dTexture)) {
        printf("D3D11 input texture creation failed\n");
        return false;
    }

    if (!d3dInput.Upload(d3dTexture.Get(), 0, d3dDevice, d3dContext, nullptr,
                         conversion, deviceOutput.data, nullptr,
                         uploadedEvent, preprocessedEvent)) {
        printf("D3D11/CUDA integration upload failed\n");
        return false;
    }

    std::vector<uint16_t> d3d(outputSamples);
    if (cudaMemcpy(d3d.data(), deviceOutput.data, outputBytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("D3D11 result download failed: %s\n",
               cudaGetErrorString(cudaGetLastError()));
        return false;
    }

    const ConversionComparison d3dComparison =
        CompareWithReference(d3d, planes, reference->strideBytes);

    printf("  %s %s %-8s %-8s: %.5f%% exact, max FP16 code delta %u\n",
           formatName, matrixName, PatternName(pattern), "D3D11",
           100.0 * d3dComparison.exact / outputSamples,
           d3dComparison.maximumDelta);
    if (d3dComparison.maximumDelta > 1) {
        printf("    D3D11 mismatch at plane %d, x=%d, y=%d\n",
               d3dComparison.maximumPlane,
               d3dComparison.maximumX,
               d3dComparison.maximumY);
        return false;
    }

    if (d3d != gpu) {
        printf("    D3D11 output differs from software output\n");
        return false;
    }

    return true;
}

auto RunFp16PlanarToRgb48Test() -> bool {
    constexpr uint16_t samples[] {
        0xBC00, // -1.0
        0x8000, // -0.0
        0x0000, //  0.0
        0x0001, // smallest positive subnormal
        0x3400, //  0.25
        0x3800, //  0.5
        0x3A00, //  0.75
        0x3BFF, // largest value below 1.0
        0x3C00, //  1.0
        0x3C01, // smallest value above 1.0
        0x4000, //  2.0
    };
    constexpr int width = 257;
    constexpr int height = 3;
    constexpr size_t pixelCount = static_cast<size_t>(width) * height;
    constexpr size_t componentCount = pixelCount * 3;

    std::vector<uint16_t> planar(componentCount);
    for (size_t i = 0; i < pixelCount; ++i) {
        for (size_t channel = 0; channel < 3; ++channel) {
            planar[channel * pixelCount + i] =
                samples[(i + 3 * channel) % std::size(samples)];
        }
    }

    std::vector<uint16_t> expected;
    std::wstring referenceError;
    if (!MLFilter::ConvertFp16PlanarToRgb48Reference(
            planar.data(), width, height, expected, referenceError)) {
        wprintf(L"zimg FP16-to-RGB48 reference failed: %ls\n", referenceError.c_str());
        return false;
    }

    DeviceBuffer devicePlanar;
    DeviceBuffer devicePacked;
    const size_t bytes = componentCount * sizeof(uint16_t);
    if (cudaMalloc(&devicePlanar.data, bytes) != cudaSuccess ||
        cudaMalloc(&devicePacked.data, bytes) != cudaSuccess ||
        cudaMemcpy(devicePlanar.data, planar.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("FP16-to-RGB48 allocation/upload failed: %s\n",
               cudaGetErrorString(cudaGetLastError()));
        return false;
    }

    if (MLFilter::LaunchFp16PlanarToRgb48(
            devicePlanar.data, devicePacked.data, static_cast<size_t>(width) * 6,
            width, height, nullptr) != cudaSuccess) {
        printf("FP16-to-RGB48 launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return false;
    }

    std::vector<uint16_t> actual(componentCount);
    if (cudaMemcpy(actual.data(), devicePacked.data, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("FP16-to-RGB48 download failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return false;
    }

    if (actual != expected) {
        for (size_t component = 0; component < componentCount; ++component) {
            if (actual[component] != expected[component]) {
                printf("FP16-to-RGB48 mismatch at component %zu: expected %u, got %u\n",
                       component, expected[component], actual[component]);
                break;
            }
        }
        return false;
    }

    constexpr size_t mappedStride = static_cast<size_t>(width) * 6 + 6;
    std::vector<uint16_t> mappedStorage(mappedStride * height / sizeof(uint16_t), 0xA55A);
    HostRegistration registration;
    if (cudaHostRegister(mappedStorage.data(), mappedStorage.size() * sizeof(uint16_t),
                         cudaHostRegisterMapped) != cudaSuccess) {
        printf("Mapped RGB48 output registration failed: %s\n",
               cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    registration.address = mappedStorage.data();

    void *mappedDevice = nullptr;
    if (cudaHostGetDevicePointer(&mappedDevice, mappedStorage.data(), 0) != cudaSuccess ||
        MLFilter::LaunchFp16PlanarToRgb48(
            devicePlanar.data, mappedDevice, mappedStride, width, height, nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        printf("Mapped FP16-to-RGB48 output failed: %s\n",
               cudaGetErrorString(cudaGetLastError()));
        return false;
    }

    std::vector<uint16_t> mappedActual(componentCount);
    const size_t rowComponents = static_cast<size_t>(width) * 3;
    for (int y = 0; y < height; ++y) {
        const auto *row = reinterpret_cast<const uint16_t *>(
            reinterpret_cast<const unsigned char *>(mappedStorage.data()) +
            static_cast<size_t>(y) * mappedStride);
        std::copy_n(row, rowComponents, mappedActual.data() + static_cast<size_t>(y) * rowComponents);
    }
    if (mappedActual != expected) {
        printf("Mapped FP16-to-RGB48 output did not match the tight device result.\n");
        return false;
    }

    printf("  FP16 planar to RGB48 device/mapped packing: exact\n");
    return true;
}

}

auto main() -> int {
    printf("MLFilter software/D3D11 GPU conversion tests\n");

    ComRef<ID3D11Device> d3dDevice;
    ComRef<ID3D11DeviceContext> d3dContext;
    if (!CreateD3D11TestDevice(d3dDevice, d3dContext)) {
        printf("Could not create a D3D11 device on the current CUDA adapter\n");
        return 1;
    }

    CudaEvent uploadedEvent;
    CudaEvent preprocessedEvent;
    if (cudaEventCreate(&uploadedEvent.value) != cudaSuccess ||
        cudaEventCreate(&preprocessedEvent.value) != cudaSuccess) {
        printf("CUDA event creation failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }

    MLFilter::D3D11CudaInput d3dInput(kWidth, kHeight);
    int failures = 0;
    int total = 0;

    ++total;
    if (!RunFp16PlanarToRgb48Test()) {
        ++failures;
    }

    const auto run = [&](MLFilter::Yuv420Format format, bool bt709, Pattern pattern) {
        ++total;
        if (!RunConversionTest(format, bt709, pattern,
                               d3dDevice.Get(), d3dContext.Get(), d3dInput,
                               uploadedEvent.value, preprocessedEvent.value)) {
            ++failures;
        }
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
