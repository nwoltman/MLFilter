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
// over real frames decoded from a video file by ffmpeg, and reports end-to-end
// throughput plus a per-stage breakdown so the inference cost is visible in isolation.
//
// The engine is built (or reused) with MLFilter's own TensorRTEngineBuilder, so the
// number reflects MLFilter's engine-build choices, not trtexec's.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <windows.h>

#include "color/yuv_to_rgb.h"
#include "engine/inference_session.h"
#include "engine/tensorrt_engine_builder.h"
#include "settings.h"

namespace {

using Clock = std::chrono::steady_clock;
using MLFilter::YuvToRgbConverter;

// ---- child-process helpers ----------------------------------------------------------

// Runs a command line to completion and returns its captured stdout, or nullopt on
// spawn failure. Used for ffprobe.
auto RunCapture(std::wstring commandLine) -> std::optional<std::string> {
    SECURITY_ATTRIBUTES sa { sizeof(sa), nullptr, TRUE };
    HANDLE readPipe = nullptr;
    HANDLE writePipe = nullptr;
    if (!CreatePipe(&readPipe, &writePipe, &sa, 0)) {
        return std::nullopt;
    }
    SetHandleInformation(readPipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si {};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = writePipe;
    si.hStdError = writePipe;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

    PROCESS_INFORMATION pi {};
    const BOOL ok = CreateProcessW(nullptr, commandLine.data(), nullptr, nullptr, TRUE,
                                   CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    CloseHandle(writePipe);
    if (!ok) {
        CloseHandle(readPipe);
        return std::nullopt;
    }

    std::string out;
    char buffer[4096];
    DWORD read = 0;
    while (ReadFile(readPipe, buffer, sizeof(buffer), &read, nullptr) && read > 0) {
        out.append(buffer, read);
    }
    CloseHandle(readPipe);
    WaitForSingleObject(pi.hProcess, INFINITE);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return out;
}

// A running ffmpeg process whose decoded rawvideo is read frame by frame from a pipe.
class FrameReader {
public:
    ~FrameReader() {
        if (_readPipe != nullptr) {
            CloseHandle(_readPipe);
        }
        if (_process != nullptr) {
            TerminateProcess(_process, 0);
            CloseHandle(_process);
        }
    }

    auto Start(std::wstring commandLine) -> bool {
        SECURITY_ATTRIBUTES sa { sizeof(sa), nullptr, TRUE };
        HANDLE writePipe = nullptr;
        // A generous pipe buffer lets ffmpeg decode ahead while we process the current frame.
        if (!CreatePipe(&_readPipe, &writePipe, &sa, 1 << 22)) {
            return false;
        }
        SetHandleInformation(_readPipe, HANDLE_FLAG_INHERIT, 0);

        STARTUPINFOW si {};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdOutput = writePipe;
        si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

        PROCESS_INFORMATION pi {};
        const BOOL ok = CreateProcessW(nullptr, commandLine.data(), nullptr, nullptr, TRUE,
                                       CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
        CloseHandle(writePipe);
        if (!ok) {
            CloseHandle(_readPipe);
            _readPipe = nullptr;
            return false;
        }
        _process = pi.hProcess;
        CloseHandle(pi.hThread);
        return true;
    }

    // Fills `dst` with exactly `bytes` of decoded data. Returns false at end of stream.
    auto ReadFrame(unsigned char *dst, size_t bytes) -> bool {
        size_t got = 0;
        while (got < bytes) {
            DWORD read = 0;
            const DWORD want = static_cast<DWORD>(std::min<size_t>(bytes - got, 1u << 20));
            if (!ReadFile(_readPipe, dst + got, want, &read, nullptr) || read == 0) {
                return false; // broken pipe / EOF
            }
            got += read;
        }
        return true;
    }

private:
    HANDLE _readPipe = nullptr;
    HANDLE _process = nullptr;
};

// ---- video probing ------------------------------------------------------------------

struct VideoInfo {
    int width = 0;
    int height = 0;
    int depth = 8;
    bool bt709 = true;
    bool fullRange = false;
};

auto FindValue(const std::string &text, const std::string &key) -> std::string {
    const std::string needle = key + "=";
    size_t pos = text.find(needle);
    if (pos == std::string::npos) {
        return {};
    }
    pos += needle.size();
    const size_t end = text.find_first_of("\r\n", pos);
    return text.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
}

auto ProbeVideo(const std::wstring &videoPath, VideoInfo &info, std::string &error) -> bool {
    std::wstring cmd = L"ffprobe -v error -select_streams v:0 -show_entries "
                       L"stream=width,height,pix_fmt,color_space,color_range "
                       L"-of default=noprint_wrappers=1 \"" + videoPath + L"\"";
    const auto output = RunCapture(cmd);
    if (!output) {
        error = "Failed to launch ffprobe. Is ffmpeg/ffprobe on PATH?";
        return false;
    }

    info.width = std::atoi(FindValue(*output, "width").c_str());
    info.height = std::atoi(FindValue(*output, "height").c_str());
    if (info.width <= 0 || info.height <= 0) {
        error = "ffprobe did not report a video resolution:\n" + *output;
        return false;
    }

    const std::string pixFmt = FindValue(*output, "pix_fmt");
    info.depth = (pixFmt.find("p10") != std::string::npos || pixFmt.find("p012") != std::string::npos ||
                  pixFmt.find("p016") != std::string::npos || pixFmt.find("p12") != std::string::npos ||
                  pixFmt.find("p16") != std::string::npos)
                     ? 10
                     : 8;

    const std::string space = FindValue(*output, "color_space");
    if (space == "bt709") {
        info.bt709 = true;
    } else if (space == "bt470bg" || space == "smpte170m" || space == "bt601") {
        info.bt709 = false;
    } else {
        info.bt709 = info.height >= 720; // unknown: guess by resolution, like the filter
    }

    const std::string range = FindValue(*output, "color_range");
    info.fullRange = (range == "pc" || range == "jpeg");

    return true;
}

// ---- argument parsing ---------------------------------------------------------------

struct Args {
    std::wstring videoPath;
    std::wstring modelPath;
    int frames = 300;
    int warmup = 10;
};

auto ParseArgs(int argc, wchar_t **argv, Args &args) -> bool {
    for (int i = 1; i < argc; ++i) {
        const std::wstring a = argv[i];
        const auto next = [&]() -> std::wstring { return i + 1 < argc ? argv[++i] : L""; };
        if (a == L"--video") {
            args.videoPath = next();
        } else if (a == L"--model") {
            args.modelPath = next();
        } else if (a == L"--frames") {
            args.frames = _wtoi(next().c_str());
        } else if (a == L"--warmup") {
            args.warmup = _wtoi(next().c_str());
        } else if (!a.empty() && a[0] != L'-' && args.videoPath.empty()) {
            args.videoPath = a; // positional video path
        } else {
            wprintf(L"Unknown argument: %ls\n", a.c_str());
            return false;
        }
    }
    return !args.videoPath.empty();
}

} // namespace

auto wmain(int argc, wchar_t **argv) -> int {
    Args args;
    if (!ParseArgs(argc, argv, args)) {
        wprintf(L"Usage: benchmark_x64.exe <video> [--model <onnx>] [--frames N] [--warmup N]\n");
        return 1;
    }

    if (!std::filesystem::exists(args.videoPath)) {
        wprintf(L"Video not found: %ls\n", args.videoPath.c_str());
        return 1;
    }

    // Resolve the ONNX model: explicit --model, else the path the filter is configured with.
    std::wstring modelPath = args.modelPath;
    if (modelPath.empty()) {
        MLFilter::Settings settings;
        settings.Load();
        modelPath = settings.modelPath;
    }
    if (modelPath.empty() || !std::filesystem::exists(modelPath)) {
        wprintf(L"No ONNX model. Pass --model <file.onnx> or configure one in MLFilter's settings.\n");
        return 1;
    }

    // Probe the source so the conversion uses the same format/matrix/range the filter would.
    VideoInfo video;
    std::string probeError;
    if (!ProbeVideo(args.videoPath, video, probeError)) {
        printf("%s\n", probeError.c_str());
        return 1;
    }

    wprintf(L"Video : %ls\n", args.videoPath.c_str());
    wprintf(L"Model : %ls\n", modelPath.c_str());
    wprintf(L"Input : %dx%d, %d-bit, matrix=%ls, range=%ls\n", video.width, video.height, video.depth,
            video.bt709 ? L"BT.709" : L"BT.601", video.fullRange ? L"full" : L"limited");

    // Build (or reuse) the engine for this resolution with MLFilter's own builder.
    MLFilter::TensorRTEngineBuilder builder;
    const MLFilter::EngineBuildRequest request { .onnxPath = modelPath, .width = video.width, .height = video.height };
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

    const YuvToRgbConverter::Kind kind = video.depth > 8 ? YuvToRgbConverter::Kind::P010 : YuvToRgbConverter::Kind::NV12;
    const YuvToRgbConverter::Params params {
        .kind = kind,
        .width = session->InputWidth(),
        .height = session->InputHeight(),
        .bt709 = video.bt709,
        .fullRange = video.fullRange,
        .bottomUp = false,
    };
    auto converter = YuvToRgbConverter::Create(params, error);
    if (!converter) {
        wprintf(L"Failed to create converter: %ls\n", error.c_str());
        return 1;
    }

    const int outW = session->OutputWidth();
    const int outH = session->OutputHeight();
    wprintf(L"Output: %dx%d (scale %.2fx)\n\n", outW, outH, static_cast<double>(outW) / video.width);

    // Decoded-frame layout from ffmpeg rawvideo: nv12 (8-bit) or p010le (>=10-bit), both 4:2:0
    // and tightly packed at the source width — exactly what YuvToRgbConverter::Deinterleave expects.
    const wchar_t *pixFmt = video.depth > 8 ? L"p010le" : L"nv12";
    const size_t sampleBytes = video.depth > 8 ? 2 : 1;
    const size_t frameBytes = static_cast<size_t>(video.width) * video.height * 3 / 2 * sampleBytes;

    std::vector<unsigned char> frame(frameBytes);
    // The engine downloads packed RGB48 directly (the fp16->RGB48 conversion runs on the GPU).
    // Tight rows: stride = outW * 3 uint16 = outW * 6 bytes.
    std::vector<uint16_t> packed(static_cast<size_t>(3) * outW * outH);
    const size_t packedStride = static_cast<size_t>(outW) * 6;

    FrameReader reader;
    std::wstring ffmpegCmd = L"ffmpeg -hide_banner -loglevel error -i \"" + args.videoPath +
                             L"\" -an -sn -map 0:v:0 -f rawvideo -pix_fmt " + pixFmt + L" -";
    if (!reader.Start(ffmpegCmd)) {
        wprintf(L"Failed to launch ffmpeg. Is it on PATH?\n");
        return 1;
    }

    // Per-stage accumulators (timed frames only).
    double convertSec = 0, inferSec = 0;
    int warmedUp = 0;
    int timed = 0;
    Clock::time_point timedStart;

    const int total = args.frames > 0 ? args.warmup + args.frames : 0; // 0 = run to EOF

    wprintf(L"Warming up (%d frames), then timing %ls...\n", args.warmup,
            args.frames > 0 ? std::to_wstring(args.frames).c_str() : L"all remaining");

    while (total == 0 || warmedUp + timed < total) {
        if (!reader.ReadFrame(frame.data(), frameBytes)) {
            break; // end of video
        }
        const bool timing = warmedUp >= args.warmup;
        if (timing && timed == 0) {
            timedStart = Clock::now();
        }

        const auto t0 = Clock::now();
        const unsigned short *rgb = converter->Convert(frame.data());
        if (rgb == nullptr) {
            wprintf(L"Conversion failed.\n");
            return 1;
        }
        const auto t1 = Clock::now();
        if (!session->Upload(rgb) || !session->Infer() || !session->Download(packed.data(), packedStride)) {
            wprintf(L"Inference failed.\n");
            return 1;
        }
        const auto t2 = Clock::now();

        if (timing) {
            convertSec += std::chrono::duration<double>(t1 - t0).count();
            inferSec += std::chrono::duration<double>(t2 - t1).count();
            ++timed;
        } else {
            ++warmedUp;
        }
    }

    if (timed == 0) {
        wprintf(L"\nNo frames timed (decoded %d during warmup). Try a longer clip or fewer warmup frames.\n", warmedUp);
        return 1;
    }

    const double wallSec = std::chrono::duration<double>(Clock::now() - timedStart).count();
    const double pipelineSec = convertSec + inferSec;
    const auto fps = [&](double sec) { return sec > 0 ? timed / sec : 0.0; };
    const auto ms = [&](double sec) { return sec / timed * 1000.0; };
    const auto pct = [&](double sec) { return pipelineSec > 0 ? sec / pipelineSec * 100.0 : 0.0; };

    wprintf(L"\n======================== Results (%d frames) ========================\n", timed);
    wprintf(L"  Stage          avg/frame       total      throughput    share\n");
    wprintf(L"  YUV->RGB       %8.3f ms   %8.3f s   %8.1f fps   %5.1f%%\n", ms(convertSec), convertSec, fps(convertSec), pct(convertSec));
    wprintf(L"  Inference      %8.3f ms   %8.3f s   %8.1f fps   %5.1f%%   <-- TensorRT (upload+infer+RGB48 convert+download)\n", ms(inferSec), inferSec, fps(inferSec), pct(inferSec));
    wprintf(L"  --------------------------------------------------------------------\n");
    wprintf(L"  Pipeline       %8.3f ms   %8.3f s   %8.1f fps   %5.1f%%   (sum of stages, serial)\n", ms(pipelineSec), pipelineSec, fps(pipelineSec), pct(pipelineSec));
    wprintf(L"  End-to-end     %8.3f ms   %8.3f s   %8.1f fps             (incl. ffmpeg decode wait)\n", ms(wallSec), wallSec, fps(wallSec));
    wprintf(L"====================================================================\n");

    return 0;
}
