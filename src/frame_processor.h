// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include <windows.h>

#include "color/packed_output.h"
#include "color/yuv420_format.h"
#include "d3d11_native_interfaces.h"
#include "debug_overlay.h"

struct IMediaSample;

namespace MLFilter {

class InferenceSession;

// One per pin connection. Uploads compact YUV and performs conversion, inference, and output
// packing on the GPU.
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

    // Converts, infers, and packs one top-down frame. A zero stride uses the sample's RGB48 pitch.
    auto Process(IMediaSample *in, IMediaSample *out, PackedOutputFormat outputFormat, int outputStride,
                 bool showDebugOverlay,
                 double previousFrameMs, uint64_t droppedFrames, double &overlayOverheadMs,
                 const D3D11DecoderState *d3d11State = nullptr) -> HRESULT;

    auto UnregisterInputBuffers() -> void;
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
};

}
