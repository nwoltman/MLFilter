// SPDX-License-Identifier: Apache-2.0
// Standalone throughput benchmark for MLFilter's GPU conversion and TensorRT pipeline.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cwchar>
#include <filesystem>
#include <string>
#include <vector>

#include "color/yuv420_format.h"
#include "engine/inference_session.h"
#include "engine/tensorrt_engine_builder.h"
#include "settings.h"

namespace {
using Clock = std::chrono::steady_clock;

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
    int frames = 300;
    int warmup = 10;
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
        } else return false;
    }
    return args.width > 0 && args.height > 0 && !(args.width & 1) && !(args.height & 1) &&
           args.frames > 0 && args.warmup >= 0;
}
}

auto wmain(int argc, wchar_t **argv) -> int {
    Args args;
    if (!ParseArgs(argc, argv, args)) {
        wprintf(L"Usage: benchmark_x64.exe [--width N] [--height N] [--format nv12|p010] "
                L"[--model file.onnx] [--frames N] [--warmup N]\n");
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

    const int outW = session->OutputWidth();
    const int outH = session->OutputHeight();
    std::vector<uint16_t> output(static_cast<size_t>(3) * outW * outH);
    const MLFilter::Yuv420Conversion conversion {
        .format = args.format, .bt709 = args.height >= 720, .fullRange = false
    };

    double pipelineSec = 0;
    double uploadSec = 0, preprocessSec = 0, inferenceSec = 0, packSec = 0;
    double downloadSec = 0, outputRegistrationSec = 0;
    int warmed = 0, timed = 0;
    while (warmed + timed < args.warmup + args.frames) {
        const bool timing = warmed >= args.warmup;
        const auto begin = Clock::now();
        if (!session->UploadYuv420(frame.data(), conversion) || !session->Infer() ||
            !session->Download(output.data(), static_cast<size_t>(outW) * 6,
                               output.size() * sizeof(uint16_t))) {
            wprintf(L"Inference failed.\n");
            return 1;
        }
        const auto end = Clock::now();
        if (timing) {
            pipelineSec += std::chrono::duration<double>(end - begin).count();
            MLFilter::InferenceSession::GpuStageTimings gpu;
            if (session->LastGpuTimings(gpu)) {
                uploadSec += gpu.uploadMs / 1000.0;
                preprocessSec += gpu.preprocessMs / 1000.0;
                inferenceSec += gpu.inferenceMs / 1000.0;
                packSec += gpu.packMs / 1000.0;
                downloadSec += gpu.downloadMs / 1000.0;
                outputRegistrationSec += gpu.outputRegistrationMs / 1000.0;
            }
            ++timed;
        } else ++warmed;
    }

    const auto ms = [&](double seconds) { return seconds / timed * 1000.0; };
    const auto fps = [&](double seconds) { return seconds > 0 ? timed / seconds : 0.0; };
    const auto pct = [&](double seconds) {
        return pipelineSec > 0 ? seconds / pipelineSec * 100.0 : 0.0;
    };
    wprintf(L"ONNX model: %ls\n", modelPath.c_str());
    wprintf(L"Input: %dx%d %ls, %ls limited\n", args.width, args.height,
            depth == 10 ? L"P010" : L"NV12", conversion.bt709 ? L"BT.709" : L"BT.601");
    wprintf(L"Output: %dx%d\n\n", outW, outH);
    wprintf(L"======================== Results (%d frames) ========================\n", timed);
    wprintf(L"  Stage           avg/frame     share    work\n");
    wprintf(L"  Upload         %8.3f ms    %5.1f%%    H2D (move frame to GPU)\n",
            ms(uploadSec), pct(uploadSec));
    wprintf(L"  Pre-process    %8.3f ms    %5.1f%%    Unpack/normalize/chroma/YUV-to-RGB/FP16/clamp\n",
            ms(preprocessSec), pct(preprocessSec));
    wprintf(L"  Inference      %8.3f ms    %5.1f%%    TensorRT engine inference\n",
            ms(inferenceSec), pct(inferenceSec));
    wprintf(L"  FP16 to RGB48  %8.3f ms    %5.1f%%    Planar FP16 to packed RGB48\n",
            ms(packSec), pct(packSec));
    wprintf(L"  Download       %8.3f ms    %5.1f%%    D2H (get frame from GPU)\n",
            ms(downloadSec), pct(downloadSec));
    wprintf(L"  Pin/unpin      %8.3f ms    %5.1f%%    Register output for direct DMA\n",
            ms(outputRegistrationSec), pct(outputRegistrationSec));
    wprintf(L"  --------------------------------------------------------------------------\n");
    wprintf(L"  Pipeline       %8.3f ms   %8.1f fps\n",
            ms(pipelineSec), fps(pipelineSec));
    wprintf(L"====================================================================\n");
    return 0;
}
