// SPDX-License-Identifier: Apache-2.0
// Standalone throughput benchmark for MLFilter's GPU conversion and TensorRT pipeline.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cwchar>
#include <filesystem>
#include <string>
#include <vector>

#include <windows.h>

#include <cuda_d3d11_interop.h>
#include <d3d11.h>
#include <dxgi.h>

#include "color/yuv420_format.h"
#include "engine/inference_session.h"
#include "engine/tensorrt_engine_builder.h"
#include "settings.h"

namespace {
using Clock = std::chrono::steady_clock;

template <typename T>
class ComRef {
public:
    ~ComRef() {
        Reset();
    }

    ComRef() = default;
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

enum class UploadMode {
    Software,
    D3D11,
};

auto PipelineColor(double milliseconds) -> WORD {
    if (milliseconds > 36.0) {
        return FOREGROUND_RED | FOREGROUND_INTENSITY;
    }
    if (milliseconds > 34.0) {
        return FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY;
    }
    return FOREGROUND_GREEN | FOREGROUND_INTENSITY;
}

void PrintPipelineResult(double milliseconds, double fps) {
    wprintf(L"  Pipeline       ");

    const HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_SCREEN_BUFFER_INFO consoleInfo {};
    const bool hasConsole = console != nullptr && console != INVALID_HANDLE_VALUE &&
                            GetConsoleScreenBufferInfo(console, &consoleInfo);
    const WORD foregroundMask = FOREGROUND_RED | FOREGROUND_GREEN |
                                FOREGROUND_BLUE | FOREGROUND_INTENSITY;
    const WORD color = (consoleInfo.wAttributes & ~foregroundMask) |
                       PipelineColor(milliseconds);

    if (hasConsole) {
        fflush(stdout);
    }

    const bool colorChanged = hasConsole && SetConsoleTextAttribute(console, color);

    wprintf(L"%8.3f ms  %8.1f fps", milliseconds, fps);

    if (colorChanged) {
        fflush(stdout);
        SetConsoleTextAttribute(console, consoleInfo.wAttributes);
    }

    wprintf(L"\n");
}

auto ScaleCoordinate(size_t coordinate, size_t extent, int minimum, int maximum) -> uint16_t {
    if (extent <= 1) return static_cast<uint16_t>((minimum + maximum) / 2);
    return static_cast<uint16_t>(minimum +
        (static_cast<uint64_t>(coordinate) * (maximum - minimum) + (extent - 1) / 2) /
        (extent - 1));
}

template <typename Sample>
void GenerateGradient(std::vector<unsigned char> &frame, int width, int height, int depth) {
    auto *samples = reinterpret_cast<Sample *>(frame.data());
    const int shift = depth == 10 ? 6 : 0;
    const int yMin = depth == 10 ? 64 : 16;
    const int yMax = depth == 10 ? 940 : 235;
    const int cMax = depth == 10 ? 960 : 240;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t i = static_cast<size_t>(y) * width + x;
            samples[i] = static_cast<Sample>(
                ScaleCoordinate(i, static_cast<size_t>(width) * height, yMin, yMax) << shift);
        }
    }
    Sample *uv = samples + static_cast<size_t>(width) * height;
    for (int y = 0; y < height / 2; ++y) {
        const uint16_t cr = ScaleCoordinate(y, height / 2, yMin, cMax);
        for (int x = 0; x < width / 2; ++x) {
            const size_t i = static_cast<size_t>(y) * width + x * 2;
            uv[i] = static_cast<Sample>(ScaleCoordinate(x, width / 2, yMin, cMax) << shift);
            uv[i + 1] = static_cast<Sample>(cr << shift);
        }
    }
}

struct Args {
    std::wstring modelPath;
    int width = 1920;
    int height = 1080;
    MLFilter::Yuv420Format format = MLFilter::Yuv420Format::NV12;
    int frames = 400;
    int warmup = 24;
    UploadMode uploadMode = UploadMode::Software;
};

auto ParseArgs(int argc, wchar_t **argv, Args &args) -> bool {
    for (int i = 1; i < argc; ++i) {
        const std::wstring a = argv[i];
        const auto next = [&]() -> std::wstring { return i + 1 < argc ? argv[++i] : L""; };
        if (a == L"--model") args.modelPath = next();
        else if (a == L"--width") args.width = _wtoi(next().c_str());
        else if (a == L"--height") args.height = _wtoi(next().c_str());
        else if (a == L"--frames") args.frames = _wtoi(next().c_str());
        else if (a == L"--warmup") args.warmup = _wtoi(next().c_str());
        else if (a == L"--format") {
            const auto value = next();
            if (_wcsicmp(value.c_str(), L"nv12") == 0) args.format = MLFilter::Yuv420Format::NV12;
            else if (_wcsicmp(value.c_str(), L"p010") == 0) args.format = MLFilter::Yuv420Format::P010;
            else return false;
        } else if (a == L"--upload") {
            const auto value = next();
            if (_wcsicmp(value.c_str(), L"software") == 0) args.uploadMode = UploadMode::Software;
            else if (_wcsicmp(value.c_str(), L"d3d11") == 0) args.uploadMode = UploadMode::D3D11;
            else return false;
        } else return false;
    }
    return args.width > 0 && args.height > 0 && !(args.width & 1) && !(args.height & 1) &&
           args.frames > 0 && args.warmup >= 0;
}

auto FindCudaDxgiAdapter(IDXGIAdapter **adapter) -> bool {
    *adapter = nullptr;

    int currentCudaDevice = 0;
    if (cudaGetDevice(&currentCudaDevice) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    ComRef<IDXGIFactory> factory;
    if (FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory), reinterpret_cast<void **>(factory.Address())))) {
        return false;
    }

    for (UINT index = 0;; ++index) {
        IDXGIAdapter *candidate = nullptr;
        if (factory.Get()->EnumAdapters(index, &candidate) == DXGI_ERROR_NOT_FOUND) {
            break;
        }

        int adapterCudaDevice = -1;
        const cudaError_t cudaResult = cudaD3D11GetDevice(&adapterCudaDevice, candidate);
        if (cudaResult == cudaSuccess && adapterCudaDevice == currentCudaDevice) {
            *adapter = candidate;
            return true;
        }

        if (cudaResult != cudaSuccess) {
            cudaGetLastError();
        }
        candidate->Release();
    }

    return false;
}

auto CreateD3D11BenchmarkDevice(ComRef<ID3D11Device> &device,
                                ComRef<ID3D11DeviceContext> &context,
                                std::wstring &error) -> bool {
    ComRef<IDXGIAdapter> adapter;
    if (!FindCudaDxgiAdapter(adapter.Address())) {
        error = L"Could not find a DXGI adapter for the current CUDA device.";
        return false;
    }

    const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_0};
    D3D_FEATURE_LEVEL createdLevel = D3D_FEATURE_LEVEL_11_0;
    const HRESULT hr = D3D11CreateDevice(adapter.Get(),
                                         D3D_DRIVER_TYPE_UNKNOWN,
                                         nullptr,
                                         D3D11_CREATE_DEVICE_SINGLETHREADED,
                                         levels,
                                         ARRAYSIZE(levels),
                                         D3D11_SDK_VERSION,
                                         device.Address(),
                                         &createdLevel,
                                         context.Address());
    if (FAILED(hr)) {
        error = L"Could not create a D3D11 device on the CUDA adapter.";
        return false;
    }

    return true;
}

auto CreateD3D11InputTexture(ID3D11Device *device,
                             const std::vector<unsigned char> &frame,
                             const Args &args,
                             ComRef<ID3D11Texture2D> &texture,
                             std::wstring &error) -> bool {
    const size_t sampleBytes = args.format == MLFilter::Yuv420Format::P010 ? 2 : 1;

    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = static_cast<UINT>(args.width);
    desc.Height = static_cast<UINT>(args.height);
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = args.format == MLFilter::Yuv420Format::P010 ? DXGI_FORMAT_P010 : DXGI_FORMAT_NV12;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    D3D11_SUBRESOURCE_DATA initial {};
    initial.pSysMem = frame.data();
    initial.SysMemPitch = static_cast<UINT>(static_cast<size_t>(args.width) * sampleBytes);
    initial.SysMemSlicePitch = static_cast<UINT>(frame.size());

    const HRESULT hr = device->CreateTexture2D(&desc, &initial, texture.Address());
    if (FAILED(hr)) {
        error = L"Could not create the synthetic D3D11 NV12/P010 input texture.";
        return false;
    }

    return true;
}
}

auto wmain(int argc, wchar_t **argv) -> int {
    Args args;
    if (!ParseArgs(argc, argv, args)) {
        wprintf(L"Usage: benchmark_x64.exe [--width N] [--height N] [--format nv12|p010] "
                L"[--upload software|d3d11] [--model file.onnx] [--frames N] [--warmup N]\n");
        return 1;
    }
    std::wstring modelPath = args.modelPath;
    if (modelPath.empty()) {
        MLFilter::Settings settings;
        settings.Load();
        modelPath = settings.hdModelPath;
    }
    if (modelPath.empty() || !std::filesystem::exists(modelPath)) {
        wprintf(L"No ONNX model was configured.\n");
        return 1;
    }

    MLFilter::TensorRTEngineBuilder builder;
    const MLFilter::EngineBuildRequest request {
        .onnxPath = modelPath, .width = args.width, .height = args.height
    };
    if (!builder.Exists(request)) {
        const auto result = builder.Build(request, [](const std::wstring &message) {
            wprintf(L"  %ls\n", message.c_str());
        });
        if (!result.success) {
            wprintf(L"Engine build failed: %ls\n", result.message.c_str());
            return 1;
        }
    }
    std::wstring error;
    auto session = MLFilter::InferenceSession::Create(builder.EnginePath(request), error);
    if (!session) {
        wprintf(L"Failed to load engine: %ls\n", error.c_str());
        return 1;
    }

    const int depth = args.format == MLFilter::Yuv420Format::P010 ? 10 : 8;
    const size_t frameBytes = static_cast<size_t>(args.width) * args.height * 3 / 2 *
                              (depth == 10 ? 2 : 1);
    std::vector<unsigned char> frame(frameBytes);
    if (depth == 10) GenerateGradient<uint16_t>(frame, args.width, args.height, depth);
    else GenerateGradient<uint8_t>(frame, args.width, args.height, depth);

    ComRef<ID3D11Device> d3d11Device;
    ComRef<ID3D11DeviceContext> d3d11Context;
    ComRef<ID3D11Texture2D> d3d11Texture;
    if (args.uploadMode == UploadMode::D3D11) {
        std::wstring d3d11Error;
        if (!CreateD3D11BenchmarkDevice(d3d11Device, d3d11Context, d3d11Error) ||
            !CreateD3D11InputTexture(d3d11Device.Get(), frame, args, d3d11Texture, d3d11Error)) {
            wprintf(L"D3D11 benchmark setup failed: %ls\n", d3d11Error.c_str());
            return 1;
        }
    }

    const int outW = session->OutputWidth();
    const int outH = session->OutputHeight();
    std::vector<uint16_t> output(static_cast<size_t>(3) * outW * outH);
    const MLFilter::Yuv420Conversion conversion {
        .format = args.format, .bt709 = args.height >= 720, .fullRange = false
    };

    wprintf(L"ONNX model: %ls\n", modelPath.c_str());
    wprintf(L"Input: %dx%d %ls, %ls limited\n", args.width, args.height,
            depth == 10 ? L"P010" : L"NV12", conversion.bt709 ? L"BT.709" : L"BT.601");
    wprintf(L"Upload input: %ls\n",
            args.uploadMode == UploadMode::D3D11 ? L"D3D11 texture -> CUDA buffer" : L"software frame -> CUDA buffer");
    wprintf(L"Output: %dx%d\n\n", outW, outH);
    fflush(stdout);

    double pipelineSec = 0;
#ifdef MLFILTER_ENABLE_STAGE_TIMINGS
    double uploadSec = 0, preprocessSec = 0, inferenceSec = 0, outputSec = 0;
#endif
    int warmed = 0, timed = 0;
    while (warmed + timed < args.warmup + args.frames) {
        const bool timing = warmed >= args.warmup;
        const auto begin = Clock::now();
        const bool uploaded = args.uploadMode == UploadMode::D3D11
            ? session->UploadD3D11Yuv420(d3d11Texture.Get(), 0, d3d11Device.Get(),
                                         d3d11Context.Get(), nullptr, conversion)
            : session->UploadYuv420(frame.data(), frame.size(), conversion);

        if (!uploaded || !session->Infer() ||
            !session->Download(output.data(), static_cast<size_t>(outW) * 6,
                               output.size() * sizeof(uint16_t))) {
            wprintf(L"Inference failed.\n");
            return 1;
        }
        const auto end = Clock::now();
        if (timing) {
            pipelineSec += std::chrono::duration<double>(end - begin).count();
#ifdef MLFILTER_ENABLE_STAGE_TIMINGS
            MLFilter::InferenceSession::GpuStageTimings gpu;
            if (session->LastGpuTimings(gpu)) {
                uploadSec += gpu.uploadMs / 1000.0;
                preprocessSec += gpu.preprocessMs / 1000.0;
                inferenceSec += gpu.inferenceMs / 1000.0;
                outputSec += gpu.outputMs / 1000.0;
            }
#endif
            ++timed;
        } else ++warmed;
    }

    const auto ms = [&](double seconds) { return seconds / timed * 1000.0; };
    const auto fps = [&](double seconds) { return seconds > 0 ? timed / seconds : 0.0; };

    wprintf(L"======================== Results (%d frames) ========================\n", timed);
#ifdef MLFILTER_ENABLE_STAGE_TIMINGS
    const auto pct = [&](double seconds) {
        return pipelineSec > 0 ? seconds / pipelineSec * 100.0 : 0.0;
    };
    wprintf(L"  Stage           avg/frame     share    work\n");
    wprintf(L"  Upload         %8.3f ms    %5.1f%%    %ls\n",
            ms(uploadSec), pct(uploadSec),
            args.uploadMode == UploadMode::D3D11 ? L"D3D11 texture interop/pack to CUDA buffer" : L"H2D (move frame to GPU)");
    wprintf(L"  Pre-process    %8.3f ms    %5.1f%%    Unpack/normalize/chroma/YUV-to-RGB/FP16/clamp\n",
            ms(preprocessSec), pct(preprocessSec));
    wprintf(L"  Inference      %8.3f ms    %5.1f%%    TensorRT engine inference\n",
            ms(inferenceSec), pct(inferenceSec));
    wprintf(L"  RGB48 output   %8.3f ms    %5.1f%%    FP16 packing directly to mapped host memory\n",
            ms(outputSec), pct(outputSec));
    wprintf(L"  --------------------------------------------------------------------------\n");
#endif
    PrintPipelineResult(ms(pipelineSec), fps(pipelineSec));
    wprintf(L"====================================================================\n");

    // The benchmark owns its output vector, so unregister it before the vector is destroyed.
    session->UnregisterOutputBuffers();
    return 0;
}
