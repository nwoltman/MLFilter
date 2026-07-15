// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <windows.h>

#include "color/yuv420_format.h"
#include "d3d11_native_interfaces.h"
#include "debug_overlay.h"

struct IMediaSample;

namespace MLFilter {

class InferenceSession;

// One per pin connection. Uploads compact YUV and performs conversion, inference, and RGB48
// packing on the GPU before copying the result into the output sample.
class FrameProcessor {
public:
    // Builds the session (deserializing the engine) and the converter for the given input
    // format. Returns nullptr on failure; error receives the reason.
    static auto Create(const std::filesystem::path &enginePath,
                       Yuv420Format format,
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
    auto Process(IMediaSample *in, IMediaSample *out, bool showDebugOverlay,
                 double previousFrameMs, double &overlayOverheadMs,
                 const D3D11DecoderState *d3d11State = nullptr) -> HRESULT;

    auto UnregisterOutputBuffers() -> void;

private:
    enum class TransportMode {
        HostCopy,
        D3D11Native,
    };

    FrameProcessor() = default;

    std::unique_ptr<InferenceSession> _session;
    DebugOverlay _debugOverlay;
    Yuv420Conversion _conversion {};
    std::string _hostTransportLine;
    std::string _nativeTransportLine;
    TransportMode _transportMode = TransportMode::HostCopy;

    int _outW = 0;
    int _outH = 0;
    int _rowBytes = 0;
    size_t _tightOutputBytes = 0;
};

}
