// SPDX-License-Identifier: Apache-2.0

#include "frame_processor.h"

#include <streams.h>

#include "formats.h"
#include "engine/inference_session.h"

namespace MLFilter {

auto FrameProcessor::Create(const std::filesystem::path &enginePath,
                            YuvToRgbConverter::Kind kind,
                            bool bt709,
                            bool fullRange,
                            std::wstring &error) -> std::unique_ptr<FrameProcessor> {
    auto processor = std::unique_ptr<FrameProcessor>(new FrameProcessor());

    processor->_session = InferenceSession::Create(enginePath, error);
    if (!processor->_session) {
        return nullptr;
    }

    const YuvToRgbConverter::Params params {
        .kind = kind,
        .width = processor->_session->InputWidth(),
        .height = processor->_session->InputHeight(),
        .bt709 = bt709,
        .fullRange = fullRange,
    };
    processor->_converter = YuvToRgbConverter::Create(params, error);
    if (!processor->_converter) {
        return nullptr;
    }

    processor->_outW = processor->_session->OutputWidth();
    processor->_outH = processor->_session->OutputHeight();

    return processor;
}

FrameProcessor::~FrameProcessor() = default;

auto FrameProcessor::Process(IMediaSample *in, IMediaSample *out) -> HRESULT {
    BYTE *srcBuffer = nullptr;
    if (FAILED(in->GetPointer(&srcBuffer)) || srcBuffer == nullptr) {
        return E_FAIL;
    }

    const YuvToRgbConverter::PlanarRgbFp16 *rgb = _converter->Convert(srcBuffer);
    if (rgb == nullptr) {
        return E_FAIL;
    }

    if (!_session->Upload(rgb->r, rgb->g, rgb->b, static_cast<size_t>(rgb->strideBytes)) || !_session->Infer()) {
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
    if (!_session->Download(dstBuffer, static_cast<size_t>(stride))) {
        return E_FAIL;
    }

    out->SetActualDataLength(stride * _outH);
    return S_OK;
}

}
