// SPDX-License-Identifier: Apache-2.0
//
// Standalone throughput benchmark for MLFilter's TensorRT inference pipeline.
//
// This is the MLFilter counterpart to mpc-vapoursynth-scripts' benchmark.vpy (which
// times the vsmlrt `core.trt.Model` pipeline via `vspipe --filter-time`). It lets the
// two TensorRT implementations be compared head-to-head outside a media player.
//
// It runs MLFilter's *actual* per-frame pipeline — the same code FrameProcessor uses
// during playback:
//   1. zimg YUV -> planar RGB fp16 conversion (YuvToRgbConverter)
//   2. the TensorRT engine: host->device upload, enqueueV3, the on-GPU fp16->RGB48 conversion
//      kernel, and the device->host copy into the (RGB48) output buffer (InferenceSession)
// over a deterministic synthetic limited-range frame, and reports throughput plus a
// per-stage breakdown so repeated runs are directly comparable.
//
// The engine is built (or reused) with MLFilter's own TensorRTEngineBuilder, so the
// number reflects MLFilter's engine-build choices, not trtexec's.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cwchar>
#include <filesystem>
#include <string>
#include <vector>

#include "color/yuv_to_rgb.h"
#include "engine/inference_session.h"
#include "engine/tensorrt_engine_builder.h"
#include "settings.h"

namespace {

using Clock = std::chrono::steady_clock;
using MLFilter::YuvToRgbConverter;

// ---- synthetic input ---------------------------------------------------------------

auto ScaleCoordinate(size_t coordinate, size_t extent, int minimum, int maximum) -> uint16_t {
    if (extent <= 1) {
        return static_cast<uint16_t>((minimum + maximum) / 2);
    }
    return static_cast<uint16_t>(
        minimum + (static_cast<uint64_t>(coordinate) * (maximum - minimum) + (extent - 1) / 2) /
                      (extent - 1));
}

template <typename Sample>
void GenerateGradient(std::vector<unsigned char> &frame, int width, int height, int bitDepth) {
    auto *samples = reinterpret_cast<Sample *>(frame.data());
    const int shift = bitDepth == 10 ? 6 : 0;
    const int yMinimum = bitDepth == 10 ? 64 : 16;
    const int yMaximum = bitDepth == 10 ? 940 : 235;
    const int cMinimum = yMinimum;
    const int cMaximum = bitDepth == 10 ? 960 : 240;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t position = static_cast<size_t>(y) * width + x;
            const uint16_t value =
                ScaleCoordinate(position, static_cast<size_t>(width) * height, yMinimum, yMaximum);
            samples[position] = static_cast<Sample>(value << shift);
        }
    }

    // NV12/P010 share one Cb/Cr pair per 2x2 block. These orthogonal sweeps cover
    // the complete legal chroma plane at the available resolution; the independent
    // luma sweep adds as many distinct limited-range colours as the frame can hold.
    Sample *uv = samples + static_cast<size_t>(width) * height;
    const int chromaWidth = width / 2;
    const int chromaHeight = height / 2;
    for (int y = 0; y < chromaHeight; ++y) {
        const uint16_t cr = ScaleCoordinate(y, chromaHeight, cMinimum, cMaximum);
        for (int x = 0; x < chromaWidth; ++x) {
            const uint16_t cb = ScaleCoordinate(x, chromaWidth, cMinimum, cMaximum);
            const size_t offset = static_cast<size_t>(y) * width + static_cast<size_t>(x) * 2;
            uv[offset] = static_cast<Sample>(cb << shift);
            uv[offset + 1] = static_cast<Sample>(cr << shift);
        }
    }
}

// ---- argument parsing ---------------------------------------------------------------

struct Args {
    std::wstring modelPath;
    int width = 1920;
    int height = 1080;
    YuvToRgbConverter::Kind kind = YuvToRgbConverter::Kind::NV12;
    int frames = 300;
    int warmup = 10;
};

auto ParseArgs(int argc, wchar_t **argv, Args &args) -> bool {
    for (int i = 1; i < argc; ++i) {
        const std::wstring a = argv[i];
        const auto next = [&]() -> std::wstring { return i + 1 < argc ? argv[++i] : L""; };
        if (a == L"--model") {
            args.modelPath = next();
        } else if (a == L"--width") {
            args.width = _wtoi(next().c_str());
        } else if (a == L"--height") {
            args.height = _wtoi(next().c_str());
        } else if (a == L"--format") {
            const std::wstring format = next();
            if (_wcsicmp(format.c_str(), L"nv12") == 0) {
                args.kind = YuvToRgbConverter::Kind::NV12;
            } else if (_wcsicmp(format.c_str(), L"p010") == 0) {
                args.kind = YuvToRgbConverter::Kind::P010;
            } else {
                wprintf(L"Unsupported format: %ls (expected nv12 or p010)\n", format.c_str());
                return false;
            }
        } else if (a == L"--frames") {
            args.frames = _wtoi(next().c_str());
        } else if (a == L"--warmup") {
            args.warmup = _wtoi(next().c_str());
        } else {
            wprintf(L"Unknown argument: %ls\n", a.c_str());
            return false;
        }
    }
    if (args.width <= 0 || args.height <= 0 || (args.width & 1) != 0 || (args.height & 1) != 0) {
        wprintf(L"Width and height must be positive even numbers for 4:2:0 input.\n");
        return false;
    }
    if (args.frames <= 0 || args.warmup < 0) {
        wprintf(L"Frames must be positive and warmup must not be negative.\n");
        return false;
    }
    return true;
}

} // namespace

auto wmain(int argc, wchar_t **argv) -> int {
    Args args;
    if (!ParseArgs(argc, argv, args)) {
        wprintf(L"Usage: benchmark_x64.exe [--width N] [--height N] [--format nv12|p010] "
                L"[--model <onnx>] [--frames N] [--warmup N]\n");
        return 1;
    }

    // Resolve the ONNX model: explicit --model, else the path the filter is configured with.
    std::wstring modelPath = args.modelPath;
    if (modelPath.empty()) {
        MLFilter::Settings settings;
        settings.Load();
        modelPath = settings.hdModelPath;
    }
    if (modelPath.empty() || !std::filesystem::exists(modelPath)) {
        wprintf(L"No ONNX model. Pass --model <file.onnx> or configure one in MLFilter's settings.\n");
        return 1;
    }

    wprintf(L"Model : %ls\n", modelPath.c_str());
    const int bitDepth = args.kind == YuvToRgbConverter::Kind::P010 ? 10 : 8;
    const wchar_t *formatName = args.kind == YuvToRgbConverter::Kind::P010 ? L"P010" : L"NV12";
    const bool bt709 = args.height >= 720;
    wprintf(L"Input : %dx%d, %ls synthetic gradient, matrix=%ls, range=limited\n",
            args.width, args.height, formatName, bt709 ? L"BT.709" : L"BT.601");

    // Build (or reuse) the engine for this resolution with MLFilter's own builder.
    MLFilter::TensorRTEngineBuilder builder;
    const MLFilter::EngineBuildRequest request { .onnxPath = modelPath, .width = args.width, .height = args.height };
    if (!builder.Exists(request)) {
        wprintf(L"Building TensorRT engine (first run for this resolution; this can take a while)...\n");
        const MLFilter::EngineBuildResult result = builder.Build(request, [](const std::wstring &msg) {
            wprintf(L"  %ls\n", msg.c_str());
        });
        if (!result.success) {
            wprintf(L"Engine build failed: %ls\n", result.message.c_str());
            return 1;
        }
    }
    const std::filesystem::path enginePath = builder.EnginePath(request);
    wprintf(L"Engine: %ls\n", enginePath.filename().wstring().c_str());

    std::wstring error;
    auto session = MLFilter::InferenceSession::Create(enginePath, error);
    if (!session) {
        wprintf(L"Failed to load engine: %ls\n", error.c_str());
        return 1;
    }

    const YuvToRgbConverter::Params params {
        .kind = args.kind,
        .width = session->InputWidth(),
        .height = session->InputHeight(),
        .bt709 = bt709,
        .fullRange = false,
    };
    auto converter = YuvToRgbConverter::Create(params, error);
    if (!converter) {
        wprintf(L"Failed to create converter: %ls\n", error.c_str());
        return 1;
    }

    const int outW = session->OutputWidth();
    const int outH = session->OutputHeight();
    wprintf(L"Output: %dx%d (scale %.2fx)\n\n", outW, outH, static_cast<double>(outW) / args.width);

    // Tightly packed 4:2:0 input, matching YuvToRgbConverter::Deinterleave.
    const size_t sampleBytes = bitDepth == 10 ? 2 : 1;
    const size_t frameBytes = static_cast<size_t>(args.width) * args.height * 3 / 2 * sampleBytes;

    std::vector<unsigned char> frame(frameBytes);
    if (bitDepth == 10) {
        GenerateGradient<uint16_t>(frame, args.width, args.height, bitDepth);
    } else {
        GenerateGradient<uint8_t>(frame, args.width, args.height, bitDepth);
    }
    // The engine downloads packed RGB48 directly (the fp16->RGB48 conversion runs on the GPU).
    // Tight rows: stride = outW * 3 uint16 = outW * 6 bytes.
    std::vector<uint16_t> packed(static_cast<size_t>(3) * outW * outH);
    const size_t packedStride = static_cast<size_t>(outW) * 6;

    // Per-stage accumulators (timed frames only). The convert stage is broken down further into
    // its CPU sub-passes (deinterleave -> zimg) so the cost is visible. The inference stage is
    // broken down (via on-stream CUDA events) into upload / GPU compute / download so the
    // overlappable transfer cost is separable from the serial compute cost (sizing multi-stream).
    double convertSec = 0, inferSec = 0;
    double deinterleaveSec = 0, zimgSec = 0;
    double uploadSec = 0, computeSec = 0, downloadSec = 0;
    int warmedUp = 0;
    int timed = 0;
    const int total = args.warmup + args.frames;

    wprintf(L"Warming up (%d frames), then timing %d...\n", args.warmup, args.frames);

    while (warmedUp + timed < total) {
        const bool timing = warmedUp >= args.warmup;
        const auto t0 = Clock::now();
        YuvToRgbConverter::StageTimings stage;
        const YuvToRgbConverter::PlanarRgbFp16 *rgb = converter->Convert(frame.data(), timing ? &stage : nullptr);
        if (rgb == nullptr) {
            wprintf(L"Conversion failed.\n");
            return 1;
        }
        const auto t1 = Clock::now();
        if (!session->Upload(rgb->r, rgb->g, rgb->b, static_cast<size_t>(rgb->strideBytes)) ||
            !session->Infer() || !session->Download(packed.data(), packedStride)) {
            wprintf(L"Inference failed.\n");
            return 1;
        }
        const auto t2 = Clock::now();

        if (timing) {
            convertSec += std::chrono::duration<double>(t1 - t0).count();
            inferSec += std::chrono::duration<double>(t2 - t1).count();
            deinterleaveSec += stage.deinterleaveSec;
            zimgSec += stage.zimgSec;
            MLFilter::InferenceSession::GpuStageTimings gpu;
            if (session->LastGpuTimings(gpu)) {
                uploadSec += gpu.uploadMs / 1000.0;
                computeSec += gpu.computeMs / 1000.0;
                downloadSec += gpu.downloadMs / 1000.0;
            }
            ++timed;
        } else {
            ++warmedUp;
        }
    }

    if (timed == 0) {
        wprintf(L"\nNo frames timed.\n");
        return 1;
    }

    const double pipelineSec = convertSec + inferSec;
    const auto fps = [&](double sec) { return sec > 0 ? timed / sec : 0.0; };
    const auto ms = [&](double sec) { return sec / timed * 1000.0; };
    const auto pct = [&](double sec) { return pipelineSec > 0 ? sec / pipelineSec * 100.0 : 0.0; };

    wprintf(L"\n======================== Results (%d frames) ========================\n", timed);
    wprintf(L"  Stage          avg/frame       total      throughput    share\n");
    wprintf(L"  YUV->RGB       %8.3f ms   %8.3f s   %8.1f fps   %5.1f%%\n", ms(convertSec), convertSec, fps(convertSec), pct(convertSec));
    wprintf(L"    deinterleave   %8.3f ms   %8.3f s   %8.1f fps   %5.1f%%   (CPU unpack to planar)\n", ms(deinterleaveSec), deinterleaveSec, fps(deinterleaveSec), pct(deinterleaveSec));
    wprintf(L"    zimg           %8.3f ms   %8.3f s   %8.1f fps   %5.1f%%   (CPU upsample+matrix+fp16)\n", ms(zimgSec), zimgSec, fps(zimgSec), pct(zimgSec));
    wprintf(L"  Inference      %8.3f ms   %8.3f s   %8.1f fps   %5.1f%%   <-- TensorRT\n", ms(inferSec), inferSec, fps(inferSec), pct(inferSec));
    wprintf(L"    upload         %8.3f ms   %8.3f s   %8.1f fps   %5.1f%%   (GPU: H2D copy + clamp)\n", ms(uploadSec), uploadSec, fps(uploadSec), pct(uploadSec));
    wprintf(L"    compute        %8.3f ms   %8.3f s   %8.1f fps   %5.1f%%   (GPU: enqueueV3 + RGB48 kernel)\n", ms(computeSec), computeSec, fps(computeSec), pct(computeSec));
    wprintf(L"    download       %8.3f ms   %8.3f s   %8.1f fps   %5.1f%%   (GPU: D2H copy)\n", ms(downloadSec), downloadSec, fps(downloadSec), pct(downloadSec));
    wprintf(L"  --------------------------------------------------------------------\n");
    wprintf(L"  Pipeline       %8.3f ms   %8.3f s   %8.1f fps   %5.1f%%   (sum of stages, serial)\n", ms(pipelineSec), pipelineSec, fps(pipelineSec), pct(pipelineSec));
    wprintf(L"====================================================================\n");

    // Multi-stream / multi-thread ceiling: with enough streams + threads, the CPU convert, the two
    // copy engines (H2D / D2H) and the GPU compute all overlap, so steady-state per-frame time is
    // bounded by the single busiest resource. This is the best Change B (+ hiding the CPU convert)
    // could reach with perfect overlap — the real result lands below it (scheduling, no double copy
    // engine guaranteed, TRT context overhead).
    const double convertMs = ms(convertSec);
    const double uploadMs = ms(uploadSec);
    const double computeMs = ms(computeSec);
    const double downloadMs = ms(downloadSec);
    // (std::max with an initializer list collides with windows.h's max macro; fold by hand.)
    double bottleneckMs = convertMs;
    if (uploadMs > bottleneckMs) bottleneckMs = uploadMs;
    if (computeMs > bottleneckMs) bottleneckMs = computeMs;
    if (downloadMs > bottleneckMs) bottleneckMs = downloadMs;
    const char *bottleneck = bottleneckMs == computeMs ? "GPU compute"
                           : bottleneckMs == convertMs ? "CPU convert"
                           : bottleneckMs == downloadMs ? "D2H download"
                                                        : "H2D upload";
    wprintf(L"\n  Multi-stream ceiling (perfect overlap): bound by %hs at %.3f ms/frame = %.1f fps\n",
            bottleneck, bottleneckMs, bottleneckMs > 0 ? 1000.0 / bottleneckMs : 0.0);
    wprintf(L"  (vs current serial pipeline %.1f fps; transfer+convert overlap headroom is the gap.)\n",
            fps(pipelineSec));

    return 0;
}
