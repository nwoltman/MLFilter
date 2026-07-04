// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <windows.h>

#include "color/yuv_to_rgb.h"

struct IMediaSample;

namespace MLFilter {

class InferenceSession;

// One per pin connection. Owns the zimg converter and the TensorRT session. Process() runs a
// full frame: YUV -> RGB fp16 (CPU/zimg) -> engine + RGB48 conversion (GPU) -> copy into the
// output sample.
class FrameProcessor {
public:
    // Builds the session (deserializing the engine) and the converter for the given input
    // format. Returns nullptr on failure; error receives the reason.
    static auto Create(const std::filesystem::path &enginePath,
                       YuvToRgbConverter::Kind kind,
                       bool bt709,
                       bool fullRange,
                       std::wstring &error) -> std::unique_ptr<FrameProcessor>;

    ~FrameProcessor();

    FrameProcessor(const FrameProcessor &) = delete;
    auto operator=(const FrameProcessor &) -> FrameProcessor & = delete;

    // Engine output resolution (may differ from the input), used to negotiate the output type.
    auto OutputWidth() const -> int { return _outW; }
    auto OutputHeight() const -> int { return _outH; }

    // Converts + infers + packs one frame's pixels into out (RGB48, top-down). Does not touch
    // timestamps/flags — the caller copies those. Returns an HRESULT.
    auto Process(IMediaSample *in, IMediaSample *out) -> HRESULT;

private:
    FrameProcessor() = default;

    std::unique_ptr<InferenceSession> _session;
    std::unique_ptr<YuvToRgbConverter> _converter;

    int _outW = 0;
    int _outH = 0;
};

}
