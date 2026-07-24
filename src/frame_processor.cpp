// SPDX-License-Identifier: Apache-2.0

#include "frame_processor.h"

#include <chrono>
#include <cstdio>

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

class ComTextureRef {
public:
    ~ComTextureRef() {
        if (_texture != nullptr) {
            _texture->Release();
        }
    }

    auto Address() -> ID3D11Texture2D ** { return &_texture; }
    auto Get() const -> ID3D11Texture2D * { return _texture; }

private:
    ID3D11Texture2D *_texture = nullptr;
};

template <typename T>
class ComRef {
public:
    ~ComRef() {
        if (_object != nullptr) {
            _object->Release();
        }
    }

    auto Address() -> T ** { return &_object; }
    auto Get() const -> T * { return _object; }

private:
    T *_object = nullptr;
};

auto DxgiFormatForInput(Yuv420Format format) -> DXGI_FORMAT {
    return format == Yuv420Format::P010 ? DXGI_FORMAT_P010 : DXGI_FORMAT_NV12;
}

auto Yuv420FormatName(Yuv420Format format) -> const char * {
    return format == Yuv420Format::P010 ? "P010" : "NV12";
}

auto TransportFallbackLine(Yuv420Format format) -> std::string {
    char line[64] {};
    std::snprintf(line, sizeof(line), "TRANSPORT: HOST COPY %s", Yuv420FormatName(format));
    return line;
}

auto TransportNativeLine(Yuv420Format format) -> std::string {
    char line[64] {};
    std::snprintf(line, sizeof(line), "TRANSPORT: D3D11 NATIVE %s", Yuv420FormatName(format));
    return line;
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
    processor->_rowBytes = Rgb48Stride(processor->_outW);
    processor->_tightOutputBytes =
        static_cast<size_t>(processor->_rowBytes) * processor->_outH;
    processor->_hostTransportLine = TransportFallbackLine(format);
    processor->_nativeTransportLine = TransportNativeLine(format);
    processor->_debugOverlay.SetStreamInfo(
        OverlayFileName(enginePath),
        processor->_session->InputWidth(), processor->_session->InputHeight(),
        format == Yuv420Format::NV12 ? "NV12" : "P010",
        bt709, fullRange, processor->_outW, processor->_outH);
    processor->_debugOverlay.SetTransportInfo(processor->_hostTransportLine);

    return processor;
}

FrameProcessor::~FrameProcessor() = default;

auto FrameProcessor::Process(IMediaSample *in, IMediaSample *out, bool showDebugOverlay,
                             double previousFrameMs, uint64_t droppedFrames,
                             double &overlayOverheadMs,
                             const D3D11DecoderState *d3d11State) -> HRESULT {
    overlayOverheadMs = 0;

    bool uploaded = false;

    IMediaSampleD3D11 *d3d11Sample = nullptr;
    if (SUCCEEDED(in->QueryInterface(__uuidof(IMediaSampleD3D11),
                                     reinterpret_cast<void **>(&d3d11Sample))) &&
        d3d11Sample != nullptr) {
        ComTextureRef texture;
        UINT arraySlice = 0;
        const HRESULT textureHr = d3d11Sample->GetD3D11Texture(0, texture.Address(), &arraySlice);
        d3d11Sample->Release();

        if (FAILED(textureHr) || texture.Get() == nullptr) {
            return E_FAIL;
        }

        D3D11_TEXTURE2D_DESC desc {};
        texture.Get()->GetDesc(&desc);
        if (desc.Format != DxgiFormatForInput(_conversion.format)) {
            return VFW_E_TYPE_NOT_ACCEPTED;
        }

        ComRef<ID3D11Device> discoveredDevice;
        ID3D11Device *device = nullptr;
        if (d3d11State != nullptr && d3d11State->device != nullptr) {
            device = d3d11State->device;
        } else {
            texture.Get()->GetDevice(discoveredDevice.Address());
            device = discoveredDevice.Get();
        }
        if (device == nullptr) {
            return E_FAIL;
        }

        ComRef<ID3D11DeviceContext> discoveredContext;
        ID3D11DeviceContext *context = nullptr;
        if (d3d11State != nullptr && d3d11State->context != nullptr) {
            context = d3d11State->context;
        } else {
            device->GetImmediateContext(discoveredContext.Address());
            context = discoveredContext.Get();
        }
        if (context == nullptr) {
            return E_FAIL;
        }

        if (_session->UploadD3D11Yuv420(texture.Get(), arraySlice, device, context,
                                        d3d11State != nullptr ? d3d11State->mutex : nullptr,
                                        _conversion)) {
            if (showDebugOverlay && _transportMode != TransportMode::D3D11Native) {
                _debugOverlay.SetTransportInfo(_nativeTransportLine);
                _transportMode = TransportMode::D3D11Native;
            }
            uploaded = true;
        }

    } else {
        BYTE *srcBuffer = nullptr;
        if (FAILED(in->GetPointer(&srcBuffer)) || srcBuffer == nullptr) {
            return E_FAIL;
        }

        if (showDebugOverlay && _transportMode != TransportMode::HostCopy) {
            _debugOverlay.SetTransportInfo(_hostTransportLine);
            _transportMode = TransportMode::HostCopy;
        }
        const long inputBufferSize = in->GetSize();
        if (inputBufferSize <= 0) {
            return E_FAIL;
        }

        uploaded = _session->UploadYuv420(
            srcBuffer, static_cast<size_t>(inputBufferSize), _conversion);
    }

    if (!uploaded || !_session->Infer()) {
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
    int stride = _rowBytes;
    const long bufSize = out->GetSize();
    if (_outH > 0 && bufSize / _outH >= _rowBytes) {
        stride = static_cast<int>(bufSize / _outH);
    }

    // The engine output is converted to packed, top-down RGB48 on the GPU; this copies it into the
    // output sample, expanding each row to the allocator's row pitch.
    const size_t bufferBytes = bufSize > 0
        ? static_cast<size_t>(bufSize)
        : _tightOutputBytes;
    if (!_session->Download(dstBuffer, static_cast<size_t>(stride), bufferBytes)) {
        return E_FAIL;
    }

    if (showDebugOverlay && previousFrameMs >= 0) {
        const auto overlayBegin = std::chrono::steady_clock::now();

        const auto cache = _session->GetOutputCacheStatus();
        _debugOverlay.Draw(dstBuffer, static_cast<size_t>(stride), _outW, _outH,
                           {previousFrameMs, droppedFrames, cache.cached, cache.capacity,
                            cache.transientTransfers});

        const auto overlayEnd = std::chrono::steady_clock::now();
        overlayOverheadMs =
            std::chrono::duration<double, std::milli>(overlayEnd - overlayBegin).count();
    }

    out->SetActualDataLength(stride * _outH);
    return S_OK;
}

auto FrameProcessor::UnregisterInputBuffers() -> void {
    _session->UnregisterInputBuffers();
}

auto FrameProcessor::UnregisterOutputBuffers() -> void {
    _session->UnregisterOutputBuffers();
}

}
