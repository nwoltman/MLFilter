// SPDX-License-Identifier: Apache-2.0

#include "frame_processor.h"

#include <chrono>

#include <streams.h>

#include "formats.h"
#include "engine/inference_session.h"

namespace MLFilter {
namespace {

auto OverlayFileName(const std::filesystem::path &path) -> std::string {
    const std::wstring fileName = path.filename().native();
    std::string result;
    result.reserve(fileName.size());

    for (const wchar_t character : fileName) {
        result.push_back(character >= L' ' && character <= L'~'
                             ? static_cast<char>(character)
                             : '?');
    }

    return result;
}

}

auto FrameProcessor::Create(const std::filesystem::path &enginePath,
                            Yuv420Format format,
                            bool bt709,
                            bool fullRange,
                            std::wstring &error) -> std::unique_ptr<FrameProcessor> {
    auto processor = std::unique_ptr<FrameProcessor>(new FrameProcessor());

    processor->_session = InferenceSession::Create(enginePath, error);
    if (!processor->_session) {
        return nullptr;
    }

    processor->_conversion = {
        .format = format,
        .bt709 = bt709,
        .fullRange = fullRange,
    };

    processor->_outW = processor->_session->OutputWidth();
    processor->_outH = processor->_session->OutputHeight();
    processor->_debugOverlay.SetStreamInfo(
        OverlayFileName(enginePath),
        processor->_session->InputWidth(), processor->_session->InputHeight(),
        format == Yuv420Format::NV12 ? "NV12" : "P010",
        bt709, fullRange, processor->_outW, processor->_outH);

    return processor;
}

FrameProcessor::~FrameProcessor() = default;

auto FrameProcessor::Process(IMediaSample *in, IMediaSample *out, bool showDebugOverlay,
                             double previousFrameMs, double &overlayOverheadMs) -> HRESULT {
    overlayOverheadMs = 0;

    BYTE *srcBuffer = nullptr;
    if (FAILED(in->GetPointer(&srcBuffer)) || srcBuffer == nullptr) {
        return E_FAIL;
    }

    if (!_session->UploadYuv420(srcBuffer, _conversion) || !_session->Infer()) {
        return E_FAIL;
    }

    BYTE *dstBuffer = nullptr;
    if (FAILED(out->GetPointer(&dstBuffer)) || dstBuffer == nullptr) {
        return E_FAIL;
    }

    // The renderer's allocator may hand out a buffer with a padded row pitch (e.g. madVR pads
    // to a 4096-byte boundary) while the media type still advertises the unpadded width*6
    // stride. The renderer reads at its allocator pitch, so derive the real row stride from the
    // actual buffer (GetSize / height) and write rows at that pitch; fall back to width*6.
    const int rowBytes = Rgb48Stride(_outW); // valid bytes per row (width * 6)
    int stride = rowBytes;
    const long bufSize = out->GetSize();
    if (_outH > 0 && bufSize / _outH >= rowBytes) {
        stride = static_cast<int>(bufSize / _outH);
    }

    // The engine output is converted to packed, top-down RGB48 on the GPU; this copies it into the
    // output sample, expanding each row to the allocator's row pitch.
    const size_t bufferBytes = bufSize > 0
        ? static_cast<size_t>(bufSize)
        : static_cast<size_t>(stride) * _outH;
    if (!_session->Download(dstBuffer, static_cast<size_t>(stride), bufferBytes)) {
        return E_FAIL;
    }

    if (showDebugOverlay && previousFrameMs >= 0) {
        const auto overlayBegin = std::chrono::steady_clock::now();

        InferenceSession::GpuStageTimings gpu {};
        if (_session->LastGpuTimings(gpu)) {
            const auto cache = _session->GetOutputCacheStatus();
            _debugOverlay.Draw(dstBuffer, static_cast<size_t>(stride), _outW, _outH,
                               {gpu.uploadMs, gpu.preprocessMs, gpu.inferenceMs,
                                gpu.packMs, gpu.downloadMs, previousFrameMs,
                                cache.cached, cache.capacity, cache.transientTransfers,
                                cache.registrationFailures});
        }

        const auto overlayEnd = std::chrono::steady_clock::now();
        overlayOverheadMs =
            std::chrono::duration<double, std::milli>(overlayEnd - overlayBegin).count();
    }

    out->SetActualDataLength(stride * _outH);
    return S_OK;
}

auto FrameProcessor::UnregisterOutputBuffers() -> void {
    _session->UnregisterOutputBuffers();
}

}
