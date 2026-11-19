// SPDX-License-Identifier: Apache-2.0

#include "filter.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <thread>
#include <vector>

#include <dvdmedia.h>
#include <dxva.h>
#include <shlwapi.h>

#include "constants.h"
#include "formats.h"
#include "guids.h"
#include "input_pin.h"
#include "output_pin.h"
#include "progress_window.h"
#include "settings.h"
#include "engine/engine_builder.h"
#include "engine/tensorrt_engine_builder.h"

#pragma comment(lib, "shlwapi")

namespace MLFilter {

namespace {

auto DescribeVideoSubtype(const GUID &subtype) -> std::wstring {
    if (subtype == FOURCCMap(subtype.Data1)) {
        const wchar_t fourcc[] = {
            static_cast<wchar_t>((subtype.Data1 >> 0) & 0xFF),
            static_cast<wchar_t>((subtype.Data1 >> 8) & 0xFF),
            static_cast<wchar_t>((subtype.Data1 >> 16) & 0xFF),
            static_cast<wchar_t>((subtype.Data1 >> 24) & 0xFF),
            L'\0',
        };
        bool printable = true;
        for (int i = 0; i < 4; ++i) {
            printable = printable && fourcc[i] >= 0x20 && fourcc[i] <= 0x7E;
        }
        if (printable) {
            return std::wstring(L"'") + fourcc + L"'";
        }
    }

    wchar_t guid[64] = {};
    StringFromGUID2(subtype, guid, static_cast<int>(std::size(guid)));
    return guid;
}

auto Trim(std::wstring_view text) -> std::wstring {
    const auto isWs = [](wchar_t c) { return c == L' ' || c == L'\t' || c == L'\r' || c == L'\n'; };
    size_t begin = 0;
    size_t end = text.size();
    while (begin < end && isWs(text[begin])) {
        ++begin;
    }
    while (end > begin && isWs(text[end - 1])) {
        --end;
    }
    return std::wstring(text.substr(begin, end - begin));
}

// Make slash direction irrelevant: treat '/' and '\' as equivalent
auto NormalizeSeparators(std::wstring text) -> std::wstring {
    std::replace(text.begin(), text.end(), L'/', L'\\');
    return text;
}

auto FileMatchesGlobs(const std::wstring &path, const std::wstring &globsText) -> bool {
    std::vector<std::wstring> patterns;
    size_t start = 0;
    while (start <= globsText.size()) {
        const size_t newline = globsText.find_first_of(L"\r\n", start);
        const std::wstring line = Trim(globsText.substr(start, newline == std::wstring::npos ? std::wstring::npos : newline - start));
        if (!line.empty()) {
            patterns.push_back(line);
        }
        if (newline == std::wstring::npos) {
            break;
        }
        start = newline + 1;
    }

    if (patterns.empty()) {
        return true;  // no restriction configured
    }
    if (path.empty()) {
        return true;  // couldn't determine the file; don't remove ourselves
    }

    const std::wstring normalizedPath = NormalizeSeparators(path);
    for (const std::wstring &pattern : patterns) {
        if (PathMatchSpecW(normalizedPath.c_str(), NormalizeSeparators(pattern).c_str())) {
            return true;
        }
    }

    return false;
}

// Extracts the pixel width/height from a video media type (VIDEOINFOHEADER or VIDEOINFOHEADER2).
auto GetVideoResolution(const CMediaType &mt, int &width, int &height) -> bool {
    if (mt.formattype == FORMAT_VideoInfo && mt.cbFormat >= sizeof(VIDEOINFOHEADER)) {
        const auto *vih = reinterpret_cast<const VIDEOINFOHEADER *>(mt.pbFormat);
        width = vih->bmiHeader.biWidth;
        height = std::abs(vih->bmiHeader.biHeight);
        return width > 0 && height > 0;
    }
    if (mt.formattype == FORMAT_VideoInfo2 && mt.cbFormat >= sizeof(VIDEOINFOHEADER2)) {
        const auto *vih2 = reinterpret_cast<const VIDEOINFOHEADER2 *>(mt.pbFormat);
        width = vih2->bmiHeader.biWidth;
        height = std::abs(vih2->bmiHeader.biHeight);
        return width > 0 && height > 0;
    }
    return false;
}

auto SelectModel(const Settings &settings, int width, int height) -> const std::wstring & {
    if (!settings.sdModelPath.empty() &&
        (width < HD_MIN_WIDTH || height < HD_MIN_HEIGHT)) {
        return settings.sdModelPath;
    }
    return settings.hdModelPath;
}

struct InputConfig {
    Settings settings;
    int width = 0;
    int height = 0;
    std::filesystem::path modelPath;
};

auto ResolveInputConfig(const CMediaType &inputType, InputConfig &config) -> bool {
    Settings settings;
    settings.Load();

    int width = 0;
    int height = 0;
    if (!GetVideoResolution(inputType, width, height)) {
        return false;
    }

    const std::wstring &modelPath = SelectModel(settings, width, height);
    if (modelPath.empty() || !std::filesystem::exists(modelPath)) {
        return false;
    }

    config.settings = settings;
    config.width = width;
    config.height = height;
    config.modelPath = modelPath;
    return true;
}

// Maps an internally supported input subtype to the converter's format kind.
auto KindForSubtype(const GUID &s, Yuv420Format &kind) -> bool {
    if (s == FOURCCMap(Fourcc('N', 'V', '1', '2'))) { kind = Yuv420Format::NV12; }
    else if (s == FOURCCMap(Fourcc('P', '0', '1', '0'))) { kind = Yuv420Format::P010; }
    else { return false; }
    return true;
}

constexpr DWORD kColorInfoFlags = AMCONTROL_USED | AMCONTROL_COLORINFO_PRESENT;

auto ResolveInputColor(const CMediaType &type, int height) -> DXVA_ExtendedFormat {
    DXVA_ExtendedFormat color {};
    if (type.formattype == FORMAT_VideoInfo2 && type.cbFormat >= sizeof(VIDEOINFOHEADER2)) {
        const auto *video = reinterpret_cast<const VIDEOINFOHEADER2 *>(type.pbFormat);
        if ((video->dwControlFlags & kColorInfoFlags) == kColorInfoFlags) {
            color = std::bit_cast<DXVA_ExtendedFormat>(video->dwControlFlags);
        }
    }

    // Resolve the matrix supported by preprocessing, using source height for the default.
    const bool hd = height >= 720;
    switch (color.VideoTransferMatrix) {
    case DXVA_VideoTransferMatrix_BT709:
        break;
    case DXVA_VideoTransferMatrix_BT601:
    case DXVA_VideoTransferMatrix_SMPTE240M:
        color.VideoTransferMatrix = DXVA_VideoTransferMatrix_BT601;
        break;
    default:
        color.VideoTransferMatrix = hd ? DXVA_VideoTransferMatrix_BT709 : DXVA_VideoTransferMatrix_BT601;
        break;
    }

    if (color.VideoPrimaries == DXVA_VideoPrimaries_Unknown) {
        color.VideoPrimaries = hd ? DXVA_VideoPrimaries_BT709 : DXVA_VideoPrimaries_SMPTE170M;
    }
    if (color.VideoTransferFunction == DXVA_VideoTransFunc_Unknown) {
        color.VideoTransferFunction = DXVA_VideoTransFunc_22_709;
    }

    return color;
}

auto OutputColorInfo(const CMediaType &inputType, bool y410) -> DXVA_ExtendedFormat {
    int width = 0;
    int height = 0;
    GetVideoResolution(inputType, width, height);

    const auto input = ResolveInputColor(inputType, height);
    DXVA_ExtendedFormat output {};
    output.SampleFormat = kColorInfoFlags;
    output.NominalRange = y410 ? DXVA_NominalRange_16_235 : DXVA_NominalRange_0_255;
    output.VideoTransferMatrix = y410 ? input.VideoTransferMatrix : DXVA_VideoTransferMatrix_Unknown;
    output.VideoPrimaries = input.VideoPrimaries;
    output.VideoTransferFunction = input.VideoTransferFunction;
    return output;
}

auto GetY410Stride(const AM_MEDIA_TYPE &type, int width, int height,
                   const DXVA_ExtendedFormat &expectedColor) -> int {
    if (type.majortype != MEDIATYPE_Video || type.subtype != Y410Subtype() ||
        type.formattype != FORMAT_VideoInfo2 || type.cbFormat < sizeof(VIDEOINFOHEADER2) ||
        type.pbFormat == nullptr) {
        return 0;
    }

    const auto &video = *reinterpret_cast<const VIDEOINFOHEADER2 *>(type.pbFormat);
    const auto &bitmap = video.bmiHeader;
    constexpr int bytesPerPixel = PackedOutputBytesPerPixel(PackedOutputFormat::Y410);
    if (bitmap.biCompression != kY410Fourcc || bitmap.biBitCount != 32 ||
        bitmap.biHeight != height || bitmap.biWidth < width ||
        static_cast<uint64_t>(bitmap.biWidth) * height * bytesPerPixel > MAXLONG) {
        return 0;
    }

    // A renderer may use biWidth for the surface stride and rcTarget for the visible region.
    const RECT active {0, 0, width, height};
    if ((bitmap.biWidth != width || !IsRectEmpty(&video.rcTarget)) &&
        !EqualRect(&video.rcTarget, &active)) {
        return 0;
    }

    const auto color = std::bit_cast<DXVA_ExtendedFormat>(video.dwControlFlags);
    if ((video.dwControlFlags & kColorInfoFlags) != kColorInfoFlags ||
        color.NominalRange != expectedColor.NominalRange ||
        color.VideoTransferMatrix != expectedColor.VideoTransferMatrix ||
        color.VideoPrimaries != expectedColor.VideoPrimaries ||
        color.VideoTransferFunction != expectedColor.VideoTransferFunction) {
        return 0;
    }

    return bitmap.biWidth * bytesPerPixel;
}

}


CMLFilter::CMLFilter(LPUNKNOWN pUnk, HRESULT * /*phr*/)
    : CTransformFilter(FILTER_NAME_WIDE, pUnk, CLSID_MLFilter) {}

CMLFilter::~CMLFilter() {
    _hotkeyListener.Stop();
}

auto CMLFilter::InputPin() const -> MLFilterInputPin * {
    return static_cast<MLFilterInputPin *>(m_pInput);
}

auto CALLBACK CMLFilter::CreateInstance(LPUNKNOWN pUnk, HRESULT *phr) -> CUnknown * {
    auto *instance = new CMLFilter(pUnk, phr);
    if (instance == nullptr && phr != nullptr) {
        *phr = E_OUTOFMEMORY;
    }
    return instance;
}

auto STDMETHODCALLTYPE CMLFilter::NonDelegatingQueryInterface(REFIID riid, void **ppv) -> HRESULT {
    CheckPointer(ppv, E_POINTER);

    if (riid == IID_ISpecifyPropertyPages) {
        return GetInterface(static_cast<ISpecifyPropertyPages *>(this), ppv);
    }

    return CTransformFilter::NonDelegatingQueryInterface(riid, ppv);
}

auto CMLFilter::CheckInputType(const CMediaType *mtIn) -> HRESULT {
    CheckPointer(mtIn, E_POINTER);

    if (*mtIn->Type() != MEDIATYPE_Video) {
        return VFW_E_TYPE_NOT_ACCEPTED;
    }

    if (!ShouldProcessCurrentFile(*mtIn)) {
        ScheduleSelfRemoval();
        return VFW_E_TYPE_NOT_ACCEPTED;
    }

    return S_OK;
}

auto CMLFilter::CheckTransform(const CMediaType *mtIn, const CMediaType *mtOut) -> HRESULT {
    CheckPointer(mtIn, E_POINTER);
    CheckPointer(mtOut, E_POINTER);

    if (*mtIn->Type() != MEDIATYPE_Video || *mtOut->Type() != MEDIATYPE_Video) {
        return VFW_E_TYPE_NOT_ACCEPTED;
    }

    const GUID &subtype = *mtOut->Subtype();
    if (m_pOutput != nullptr && m_pOutput->IsConnected() &&
        subtype != *m_pOutput->CurrentMediaType().Subtype()) {
        return VFW_E_TYPE_NOT_ACCEPTED;
    }

    int width = 0;
    int height = 0;
    if (_processor != nullptr) {
        width = _processor->OutputWidth();
        height = _processor->OutputHeight();
    } else if (_inputFormatSupported || !GetVideoResolution(*mtIn, width, height)) {
        // Unsupported input formats use a placeholder at the input resolution until streaming.
        return VFW_E_TYPE_NOT_ACCEPTED;
    }

    if (subtype == Y410Subtype()) {
        return GetY410Stride(*mtOut, width, height, OutputColorInfo(*mtIn, true)) > 0
            ? S_OK : VFW_E_TYPE_NOT_ACCEPTED;
    }
    if (subtype != Rgb48Subtype()) {
        return VFW_E_TYPE_NOT_ACCEPTED;
    }

    int outW = 0;
    int outH = 0;
    return GetVideoResolution(*mtOut, outW, outH) && outW == width && outH == height
        ? S_OK : VFW_E_TYPE_NOT_ACCEPTED;
}

auto CMLFilter::GetMediaType(int iPosition, CMediaType *pMediaType) -> HRESULT {
    CheckPointer(pMediaType, E_POINTER);

    if (m_pInput == nullptr || !m_pInput->IsConnected()) {
        return E_UNEXPECTED;
    }
    if (iPosition < 0) {
        return E_INVALIDARG;
    }
    if (iPosition > 1) {
        return VFW_S_NO_MORE_ITEMS;
    }

    const bool y410 = iPosition == 1;
    const auto format = y410 ? PackedOutputFormat::Y410 : PackedOutputFormat::Rgb48;
    const int bytesPerPixel = PackedOutputBytesPerPixel(format);
    const CMediaType &inMt = m_pInput->CurrentMediaType();

    if (_processor == nullptr && _inputFormatSupported) {
        return VFW_E_TYPE_NOT_ACCEPTED;
    }

    int outW = 0;
    int outH = 0;
    if (_processor != nullptr) {
        outW = _processor->OutputWidth();
        outH = _processor->OutputHeight();
    } else if (!GetVideoResolution(inMt, outW, outH)) {
        return VFW_E_TYPE_NOT_ACCEPTED;
    }
    const uint64_t packedBytes = static_cast<uint64_t>(outW) * outH * bytesPerPixel;
    if (packedBytes > MAXLONG) {
        return VFW_E_TYPE_NOT_ACCEPTED;
    }
    const auto imageSize = static_cast<DWORD>(packedBytes);

    // Inherit timing and aspect ratio from the input where available.
    REFERENCE_TIME avgTimePerFrame = 0;
    DWORD aspectX = 0;
    DWORD aspectY = 0;
    if (inMt.formattype == FORMAT_VideoInfo2 && inMt.cbFormat >= sizeof(VIDEOINFOHEADER2)) {
        const auto *in = reinterpret_cast<const VIDEOINFOHEADER2 *>(inMt.pbFormat);
        avgTimePerFrame = in->AvgTimePerFrame;
        aspectX = in->dwPictAspectRatioX;
        aspectY = in->dwPictAspectRatioY;
    } else if (inMt.formattype == FORMAT_VideoInfo && inMt.cbFormat >= sizeof(VIDEOINFOHEADER)) {
        avgTimePerFrame = reinterpret_cast<const VIDEOINFOHEADER *>(inMt.pbFormat)->AvgTimePerFrame;
    }

    pMediaType->InitMediaType();
    pMediaType->SetType(&MEDIATYPE_Video);
    pMediaType->SetSubtype(y410 ? &Y410Subtype() : &Rgb48Subtype());
    pMediaType->SetFormatType(&FORMAT_VideoInfo2);
    pMediaType->SetTemporalCompression(FALSE);
    pMediaType->SetSampleSize(imageSize);

    auto *out = reinterpret_cast<VIDEOINFOHEADER2 *>(pMediaType->AllocFormatBuffer(sizeof(VIDEOINFOHEADER2)));
    if (out == nullptr) {
        return E_OUTOFMEMORY;
    }
    ZeroMemory(out, sizeof(*out));
    out->AvgTimePerFrame = avgTimePerFrame;
    out->dwPictAspectRatioX = aspectX != 0 ? aspectX : static_cast<DWORD>(outW);
    out->dwPictAspectRatioY = aspectY != 0 ? aspectY : static_cast<DWORD>(outH);

    out->dwControlFlags = std::bit_cast<DWORD>(OutputColorInfo(inMt, y410));

    // Active region = the whole frame. madVR can misread an empty rcSource/rcTarget.
    out->rcSource.left = 0;
    out->rcSource.top = 0;
    out->rcSource.right = outW;
    out->rcSource.bottom = outH;
    out->rcTarget = out->rcSource;

    out->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    out->bmiHeader.biWidth = outW;
    // Both FOURCC formats use a positive biHeight with top-down rows.
    out->bmiHeader.biHeight = outH;
    out->bmiHeader.biPlanes = 1;
    out->bmiHeader.biBitCount = static_cast<WORD>(8 * bytesPerPixel);
    out->bmiHeader.biCompression = y410 ? kY410Fourcc : kRgb48Fourcc;
    out->bmiHeader.biSizeImage = imageSize;

    return S_OK;
}

auto CMLFilter::DecideBufferSize(IMemAllocator *pAlloc, ALLOCATOR_PROPERTIES *pProps) -> HRESULT {
    CheckPointer(pAlloc, E_POINTER);
    CheckPointer(pProps, E_POINTER);

    if (m_pOutput == nullptr) {
        return E_UNEXPECTED;
    }

    // The output media type is already set during buffer negotiation (the pin may not yet
    // report IsConnected()).
    long requiredSize = m_pOutput->CurrentMediaType().GetSampleSize();
    if (_processor != nullptr) {
        const int stride = _y410Stride != 0 ? _y410Stride :
            _processor->OutputWidth() * PackedOutputBytesPerPixel(PackedOutputFormat::Rgb48);
        requiredSize = stride * _processor->OutputHeight();
    }

    if (pProps->cBuffers < 1) {
        pProps->cBuffers = 1;
    }
    if (pProps->cbBuffer < requiredSize) {
        pProps->cbBuffer = requiredSize;
    }

    ALLOCATOR_PROPERTIES actual = {};
    const HRESULT hr = pAlloc->SetProperties(pProps, &actual);
    if (FAILED(hr)) {
        return hr;
    }
    return actual.cbBuffer >= requiredSize ? S_OK : E_FAIL;
}

auto CMLFilter::GetPin(int n) -> CBasePin * {
    HRESULT hr = S_OK;

    if (m_pInput == nullptr) {
        m_pInput = new MLFilterInputPin(this, &hr);
        if (m_pInput == nullptr || FAILED(hr)) {
            delete m_pInput;
            m_pInput = nullptr;
            return nullptr;
        }

        m_pOutput = new MLFilterOutputPin(this, &hr);
        if (m_pOutput == nullptr || FAILED(hr)) {
            delete m_pOutput;
            m_pOutput = nullptr;
            delete m_pInput;
            m_pInput = nullptr;
            return nullptr;
        }
    }

    if (n == 0) {
        return m_pInput;
    }
    if (n == 1) {
        return m_pOutput;
    }
    return nullptr;
}

auto CMLFilter::Transform(IMediaSample *pIn, IMediaSample *pOut) -> HRESULT {
    const auto frameBegin = std::chrono::steady_clock::now();

    CheckPointer(pIn, E_POINTER);
    CheckPointer(pOut, E_POINTER);

    if (!_inputFormatSupported) {
        if (!_inputFormatErrorReported) {
            _inputFormatErrorReported = true;
            const std::wstring message =
                L"MLFilter cannot process this video's decoded pixel format: " +
                _inputFormatDescription +
                L"\n\nSupported input formats are NV12 and P010.\n\n"
                L"If you need support for this format, create an issue report at:\n"
                L"https://github.com/nwoltman/MLFilter/issues";
            MessageBoxW(nullptr, message.c_str(), L"MLFilter - Unsupported Video Format",
                        MB_OK | MB_ICONERROR | MB_TOPMOST | MB_SETFOREGROUND | MB_TASKMODAL);
            NotifyEvent(EC_ERRORABORT, VFW_E_TYPE_NOT_ACCEPTED, 0);
        }
        return VFW_E_TYPE_NOT_ACCEPTED;
    }

    // Carry timestamps and flags through to the output sample.
    REFERENCE_TIME tStart = 0;
    REFERENCE_TIME tEnd = 0;
    if (pIn->GetTime(&tStart, &tEnd) == S_OK) {
        pOut->SetTime(&tStart, &tEnd);
    }
    LONGLONG mStart = 0;
    LONGLONG mEnd = 0;
    if (pIn->GetMediaTime(&mStart, &mEnd) == S_OK) {
        pOut->SetMediaTime(&mStart, &mEnd);
    }
    pOut->SetSyncPoint(pIn->IsSyncPoint() == S_OK);
    pOut->SetDiscontinuity(pIn->IsDiscontinuity() == S_OK);
    pOut->SetPreroll(pIn->IsPreroll() == S_OK);

    if (_processor == nullptr) {
        ScheduleSelfRemoval();
        return VFW_E_TYPE_NOT_ACCEPTED;
    }

    // DirectShow renderers can attach a new surface pitch to an allocator sample.
    if (_y410Stride != 0) {
        AM_MEDIA_TYPE *sampleType = nullptr;
        const HRESULT typeHr = pOut->GetMediaType(&sampleType);
        if (FAILED(typeHr)) {
            return typeHr;
        }
        if (typeHr == S_OK) {
            if (sampleType == nullptr) {
                return VFW_E_INVALIDMEDIATYPE;
            }

            const int stride = GetY410Stride(*sampleType, _processor->OutputWidth(),
                _processor->OutputHeight(), OutputColorInfo(m_pInput->CurrentMediaType(), true));
            DeleteMediaType(sampleType);
            if (stride == 0) {
                return VFW_E_TYPE_NOT_ACCEPTED;
            }
            _y410Stride = stride;
        }
    }

    D3D11DecoderState d3d11State;
    if (InputPin() != nullptr) {
        d3d11State = InputPin()->D3D11State();
    }

    double overlayOverheadMs = 0;
    const auto format = _y410Stride != 0 ? PackedOutputFormat::Y410 : PackedOutputFormat::Rgb48;
    const HRESULT result = _processor->Process(
        pIn, pOut, format, _y410Stride, _debugOverlayEnabled.load(std::memory_order_relaxed), _previousFrameMs,
        _droppedFrames, overlayOverheadMs, &d3d11State);

    if (d3d11State.context != nullptr) {
        d3d11State.context->Release();
    }
    if (d3d11State.device != nullptr) {
        d3d11State.device->Release();
    }

    const auto frameEnd = std::chrono::steady_clock::now();
    const double totalFrameMs =
        std::chrono::duration<double, std::milli>(frameEnd - frameBegin).count();

    _previousFrameMs = totalFrameMs > overlayOverheadMs
        ? totalFrameMs - overlayOverheadMs
        : 0;

    return result;
}

auto CMLFilter::Receive(IMediaSample *pSample) -> HRESULT {
    CheckPointer(pSample, E_POINTER);

    if (ShouldDropLateFrame(pSample)) {
        ++_droppedFrames;
        m_bSampleSkipped = TRUE;

        if (!m_bQualityChanged) {
            NotifyEvent(EC_QUALITY_CHANGE, 0, 0);
            m_bQualityChanged = TRUE;
        }

        return S_OK;
    }

    return CTransformFilter::Receive(pSample);
}

auto CMLFilter::ShouldDropLateFrame(IMediaSample *sample) -> bool {
    if (!_isRunning.load(std::memory_order_acquire)) {
        return false;
    }

    REFERENCE_TIME frameStart = 0;
    REFERENCE_TIME frameEnd = 0;
    if (sample->GetTime(&frameStart, &frameEnd) != S_OK) {
        return false;
    }

    CRefTime streamTime;
    if (FAILED(StreamTime(streamTime))) {
        return false;
    }

    return frameEnd <= static_cast<REFERENCE_TIME>(streamTime);
}

auto STDMETHODCALLTYPE CMLFilter::Run(REFERENCE_TIME tStart) -> HRESULT {
    const HRESULT hr = CBaseFilter::Run(tStart);
    if (SUCCEEDED(hr)) {
        _isRunning.store(true, std::memory_order_release);
    }

    return hr;
}

auto STDMETHODCALLTYPE CMLFilter::Pause() -> HRESULT {
    _isRunning.store(false, std::memory_order_release);
    return CTransformFilter::Pause();
}

auto STDMETHODCALLTYPE CMLFilter::Stop() -> HRESULT {
    _isRunning.store(false, std::memory_order_release);
    return CTransformFilter::Stop();
}

auto CMLFilter::StartStreaming() -> HRESULT {
    _droppedFrames = 0;
    _previousFrameMs = -1;
    return CTransformFilter::StartStreaming();
}

auto CMLFilter::CompleteConnect(PIN_DIRECTION direction, IPin *pReceivePin) -> HRESULT {
    const HRESULT hr = CTransformFilter::CompleteConnect(direction, pReceivePin);
    if (FAILED(hr)) {
        return hr;
    }

    // The input's media type (and thus the real video resolution) is known once the upstream
    // pin connects, before the output pin negotiates and before playback starts.
    if (direction == PINDIR_INPUT) {
        const GUID &subtype = *m_pInput->CurrentMediaType().Subtype();
        _inputFormatSupported = IsSupportedInputSubtype(subtype);
        _inputFormatErrorReported = false;
        _inputFormatDescription = DescribeVideoSubtype(subtype);
        if (!_inputFormatSupported) {
            _processor.reset();
            return S_OK;
        }

        const HRESULT setupHr = SetupProcessorForInput();
        if (FAILED(setupHr)) {
            return setupHr;
        }
        if (_processor == nullptr) {
            ScheduleSelfRemoval();
        }
    } else if (direction == PINDIR_OUTPUT) {
        const CMediaType &outputType = m_pOutput->CurrentMediaType();
        if (*outputType.Subtype() == Y410Subtype()) {
            // The format and stride were validated by CheckTransform before connecting.
            const auto &video = *reinterpret_cast<const VIDEOINFOHEADER2 *>(outputType.Format());
            _y410Stride = video.bmiHeader.biWidth * PackedOutputBytesPerPixel(PackedOutputFormat::Y410);
        }

        // Register only after this instance is part of a complete playback graph. MPC-BE may
        // instantiate probe copies while enumerating filters; registering in the constructor
        // lets a probe copy claim the process-wide hotkey instead of the streaming instance.
        _hotkeyListener.Start(_debugOverlayEnabled);
    }

    return hr;
}

auto CMLFilter::BreakConnect(PIN_DIRECTION direction) -> HRESULT {
    if (direction == PINDIR_OUTPUT) {
        _y410Stride = 0;
        _hotkeyListener.Stop();
        _debugOverlayEnabled.store(false, std::memory_order_relaxed);
    }
    if (direction == PINDIR_INPUT) {
        _processor.reset();
        _inputFormatSupported = true;
        _inputFormatErrorReported = false;
        _inputFormatDescription.clear();
    }
    return CTransformFilter::BreakConnect(direction);
}

auto CMLFilter::ReleaseInputRegistrations() -> void {
    if (_processor != nullptr) {
        _processor->UnregisterInputBuffers();
    }
}

auto CMLFilter::ReleaseOutputRegistrations() -> void {
    if (_processor != nullptr) {
        _processor->UnregisterOutputBuffers();
    }
}

auto CMLFilter::SetupProcessorForInput() -> HRESULT {
    _processor.reset();

    if (m_pInput == nullptr) {
        return S_FALSE;
    }

    InputConfig config;
    if (!ResolveInputConfig(m_pInput->CurrentMediaType(), config)) {
        return S_FALSE;
    }

    TensorRTEngineBuilder builder;
    const EngineBuildRequest request { .onnxPath = config.modelPath, .width = config.width, .height = config.height };

    if (!builder.Exists(request)) {
        // Build synchronously (blocks graph completion until ready) with a responsive progress
        // window on its own thread.
        ProgressWindow window;
        window.Open(FILTER_NAME_WIDE);

        const EngineBuildResult result = builder.Build(request, [&window](const std::wstring &message) {
            window.Log(message);
        });

        if (result.success) {
            window.Log(L"Engine ready. Starting playback...");
            Sleep(600);
        } else {
            window.Log(L"Engine build failed; playback will stop.");
            window.Log(result.message);
        }

        window.Close();

        if (!result.success) {
            ReportEngineBuildFailure(result.message);
            return E_FAIL;
        }
    }

    const CMediaType &mt = m_pInput->CurrentMediaType();
    Yuv420Format kind = Yuv420Format::NV12;
    if (!KindForSubtype(*mt.Subtype(), kind)) {
        return S_FALSE;
    }
    const auto color = ResolveInputColor(mt, config.height);
    const bool bt709 = color.VideoTransferMatrix == DXVA_VideoTransferMatrix_BT709;
    const bool fullRange = color.NominalRange == DXVA_NominalRange_0_255;

    std::wstring error;
    _processor = FrameProcessor::Create(builder.EnginePath(request), kind, bt709, fullRange, error);

    if (_processor != nullptr) {
        _previousFrameMs = -1;
    }

    return _processor != nullptr ? S_OK : S_FALSE;
}

auto CMLFilter::ReportEngineBuildFailure(const std::wstring &details) -> void {
    std::wstring message =
        L"MLFilter could not build the TensorRT engine for this video.\n\n"
        L"Playback will stop.";
    if (!details.empty()) {
        message += L"\n\n";
        message += details;
    }

    MessageBoxW(nullptr, message.c_str(), L"MLFilter - Engine Build Failed",
                MB_OK | MB_ICONERROR | MB_TOPMOST | MB_SETFOREGROUND | MB_TASKMODAL);
    NotifyEvent(EC_ERRORABORT, E_FAIL, 0);
}

// Remove the filter when the selected model is unavailable, the resolution is above
// the configured limit, or the file path doesn't match the configured globs.
auto CMLFilter::ShouldProcessCurrentFile(const CMediaType &inputType) -> bool {
    InputConfig config;
    if (!ResolveInputConfig(inputType, config)) {
        return false;
    }

    if (config.settings.onlyRun1080pOrLower &&
        (config.width > MAX_INPUT_WIDTH || config.height > MAX_INPUT_HEIGHT)) {
        return false;
    }

    if (Trim(config.settings.fileGlobs).empty()) {
        return true;  // no glob filter configured
    }
    return FileMatchesGlobs(GetSourceFilePath(), config.settings.fileGlobs);
}

auto CMLFilter::GetSourceFilePath() const -> std::wstring {
    if (m_pGraph == nullptr) {
        return {};
    }

    IEnumFilters *enumFilters = nullptr;
    if (FAILED(m_pGraph->EnumFilters(&enumFilters)) || enumFilters == nullptr) {
        return {};
    }

    std::wstring path;
    IBaseFilter *filter = nullptr;
    while (path.empty() && enumFilters->Next(1, &filter, nullptr) == S_OK) {
        IFileSourceFilter *source = nullptr;
        if (SUCCEEDED(filter->QueryInterface(IID_IFileSourceFilter, reinterpret_cast<void **>(&source))) && source != nullptr) {
            LPOLESTR fileName = nullptr;
            AM_MEDIA_TYPE mediaType {};
            if (SUCCEEDED(source->GetCurFile(&fileName, &mediaType)) && fileName != nullptr) {
                path = fileName;
                CoTaskMemFree(fileName);
                FreeMediaType(mediaType);
            }
            source->Release();
        }
        filter->Release();
    }

    enumFilters->Release();
    return path;
}

auto CMLFilter::ScheduleSelfRemoval() -> void {
    if (_removalScheduled.exchange(true)) {
        return;  // already scheduled
    }

    IFilterGraph *graph = m_pGraph;
    if (graph == nullptr) {
        return;
    }

    graph->AddRef();
    AddRef();
    std::thread([this, graph]() {
        graph->RemoveFilter(this);
        graph->Release();
        Release();
    }).detach();
}

auto STDMETHODCALLTYPE CMLFilter::GetPages(CAUUID *pPages) -> HRESULT {
    CheckPointer(pPages, E_POINTER);

    pPages->cElems = 1;
    pPages->pElems = static_cast<GUID *>(CoTaskMemAlloc(sizeof(GUID)));
    if (pPages->pElems == nullptr) {
        return E_OUTOFMEMORY;
    }

    pPages->pElems[0] = CLSID_MLFilterPropSettings;
    return S_OK;
}

}
