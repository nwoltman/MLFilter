// SPDX-License-Identifier: Apache-2.0

#include "frame_processor.h"

#include <cstdint>
#include <cstring>

#include <streams.h>

#include "formats.h"
#include "engine/inference_session.h"

namespace MLFilter {

namespace {

// IEEE binary16 -> binary32. Portable (no F16C/intrinsic dependency); handles normals,
// subnormals, and inf/nan.
auto HalfToFloat(uint16_t h) -> float {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign; // +/- zero
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) { // normalize the subnormal
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FFu;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13); // inf / nan
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }

    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// fp16 [0,1]-ish -> 16-bit full-range integer. The engine output should sit in [0,1] but
// clamp defensively before scaling.
auto HalfToUnorm16(unsigned short h) -> uint16_t {
    float f = HalfToFloat(h);
    if (f < 0.0f) {
        f = 0.0f;
    } else if (f > 1.0f) {
        f = 1.0f;
    }
    return static_cast<uint16_t>(f * 65535.0f + 0.5f);
}

}

auto FrameProcessor::Create(const std::filesystem::path &enginePath,
                            YuvToRgbConverter::Kind kind,
                            bool bt709,
                            bool fullRange,
                            bool bottomUp,
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
        .bottomUp = bottomUp,
    };
    processor->_converter = YuvToRgbConverter::Create(params, error);
    if (!processor->_converter) {
        return nullptr;
    }

    processor->_outW = processor->_session->OutputWidth();
    processor->_outH = processor->_session->OutputHeight();
    processor->_download = std::make_unique<unsigned short[]>(static_cast<size_t>(3) * processor->_outW * processor->_outH);

    return processor;
}

FrameProcessor::~FrameProcessor() = default;

auto FrameProcessor::Process(IMediaSample *in, IMediaSample *out) -> HRESULT {
    BYTE *srcBuffer = nullptr;
    if (FAILED(in->GetPointer(&srcBuffer)) || srcBuffer == nullptr) {
        return E_FAIL;
    }

    const unsigned short *rgb = _converter->Convert(srcBuffer);
    if (rgb == nullptr) {
        return E_FAIL;
    }

    if (!_session->Upload(rgb) || !_session->Infer() || !_session->Download(_download.get())) {
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

    const size_t plane = static_cast<size_t>(_outW) * _outH;
    const unsigned short *r = _download.get();
    const unsigned short *g = r + plane;
    const unsigned short *b = g + plane;

    // Top-down packing (the RGB48 'RGB0' convention: positive biHeight, rows stored top-down).
    // RGB48 channel order is R, G, B per pixel (matching LAV/avisynth_filter's RGB48).
    for (int y = 0; y < _outH; ++y) {
        auto *dstRow = reinterpret_cast<uint16_t *>(dstBuffer + static_cast<ptrdiff_t>(y) * stride);
        const size_t row = static_cast<size_t>(y) * _outW;
        for (int x = 0; x < _outW; ++x) {
            const size_t i = row + x;
            dstRow[3 * x + 0] = HalfToUnorm16(r[i]);
            dstRow[3 * x + 1] = HalfToUnorm16(g[i]);
            dstRow[3 * x + 2] = HalfToUnorm16(b[i]);
        }
    }

    out->SetActualDataLength(stride * _outH);
    return S_OK;
}

}
