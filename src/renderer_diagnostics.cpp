// SPDX-License-Identifier: Apache-2.0

#include "renderer_diagnostics.h"

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <string>

#include <dvdmedia.h>
#include <dxva.h>
#include <wrl/client.h>

#include "formats.h"

namespace MLFilter {
namespace {

using Microsoft::WRL::ComPtr;

auto Utf8(const wchar_t *value) -> std::string {
    const int size = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 0) {
        return "<unavailable>";
    }

    std::string result(static_cast<size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value, -1, result.data(), size, nullptr, nullptr);
    result.pop_back();

    return result;
}

auto GuidText(const GUID &guid) -> std::string {
    wchar_t text[40] {};
    StringFromGUID2(guid, text, static_cast<int>(std::size(text)));

    return Utf8(text);
}

auto Hex(DWORD value) -> std::string {
    std::ostringstream text;
    text << "0x" << std::hex << std::uppercase << std::setw(8) << std::setfill('0') << value;

    return text.str();
}

auto SubtypeText(const GUID &guid) -> std::string {
    if (guid == MEDIASUBTYPE_RGB24) { return "RGB24 " + GuidText(guid); }
    if (guid == MEDIASUBTYPE_RGB32) { return "RGB32 " + GuidText(guid); }
    if (guid == MEDIASUBTYPE_ARGB32) { return "ARGB32 " + GuidText(guid); }
    if (guid == MEDIASUBTYPE_A2B10G10R10) { return "A2B10G10R10 " + GuidText(guid); }
    if (guid == MEDIASUBTYPE_A2R10G10B10) { return "A2R10G10B10 " + GuidText(guid); }

    if (guid == FOURCCMap(guid.Data1)) {
        std::string fourcc;
        for (unsigned shift = 0; shift < 32; shift += 8) {
            const char c = static_cast<char>((guid.Data1 >> shift) & 0xFF);
            if (c < 0x20 || c > 0x7E) {
                return GuidText(guid);
            }
            fourcc += c;
        }

        return "'" + fourcc + "' " + GuidText(guid);
    }

    return GuidText(guid);
}

auto VideoHeader(const AM_MEDIA_TYPE &type) -> const BITMAPINFOHEADER * {
    if (type.pbFormat == nullptr) {
        return nullptr;
    }
    if (type.formattype == FORMAT_VideoInfo2 && type.cbFormat >= sizeof(VIDEOINFOHEADER2)) {
        return &reinterpret_cast<const VIDEOINFOHEADER2 *>(type.pbFormat)->bmiHeader;
    }
    if (type.formattype == FORMAT_VideoInfo && type.cbFormat >= sizeof(VIDEOINFOHEADER)) {
        return &reinterpret_cast<const VIDEOINFOHEADER *>(type.pbFormat)->bmiHeader;
    }

    return nullptr;
}

auto DescribeMediaType(const AM_MEDIA_TYPE &type) -> std::string {
    std::ostringstream text;
    text << "major=" << GuidText(type.majortype) << " subtype=" << SubtypeText(type.subtype)
         << " header=" << (type.formattype == FORMAT_VideoInfo2 ? "VideoInfo2" :
                           type.formattype == FORMAT_VideoInfo ? "VideoInfo" : GuidText(type.formattype))
         << " formatBytes=" << type.cbFormat << " sampleBytes=" << type.lSampleSize;

    if (const auto *bih = VideoHeader(type)) {
        text << " dimensions=" << bih->biWidth << 'x' << bih->biHeight
             << " storageBitsPerPixel=" << bih->biBitCount
             << " compression=" << Hex(bih->biCompression) << " imageBytes=" << bih->biSizeImage;

        const ULONG headerBytes = type.formattype == FORMAT_VideoInfo2 ? sizeof(VIDEOINFOHEADER2)
                                                                     : sizeof(VIDEOINFOHEADER);
        if (bih->biCompression == BI_BITFIELDS && type.cbFormat >= headerBytes + 3 * sizeof(DWORD)) {
            const auto *masks = reinterpret_cast<const DWORD *>(bih + 1);
            text << " RGBMasks=" << Hex(masks[0]) << ',' << Hex(masks[1]) << ',' << Hex(masks[2]);
        }
    }
    if (type.pbFormat != nullptr && type.formattype == FORMAT_VideoInfo2 &&
        type.cbFormat >= sizeof(VIDEOINFOHEADER2)) {
        const auto *header = reinterpret_cast<const VIDEOINFOHEADER2 *>(type.pbFormat);
        text << " controlFlags=" << Hex(header->dwControlFlags)
             << " frameTime=" << header->AvgTimePerFrame;
    }

    return text.str();
}

auto OpenLog() -> std::ofstream {
    const DWORD size = GetEnvironmentVariableW(L"LOCALAPPDATA", nullptr, 0);
    if (size == 0) {
        return {};
    }

    std::wstring localAppData(size, L'\0');
    const DWORD copied = GetEnvironmentVariableW(L"LOCALAPPDATA", localAppData.data(), size);
    if (copied == 0 || copied >= size) {
        return {};
    }
    localAppData.resize(copied);

    const auto folder = std::filesystem::path(localAppData) / L"MLFilter" / L"Logs";
    std::error_code error;
    std::filesystem::create_directories(folder, error);
    if (error) {
        return {};
    }

    const auto path = folder / (L"renderer-formats-" + std::to_wstring(GetCurrentProcessId()) + L".log");
    return std::ofstream(path, std::ios::app | std::ios::binary);
}

auto LogIdentity(std::ostream &log, IPin *pin) -> void {
    PIN_INFO info {};
    const HRESULT result = pin->QueryPinInfo(&info);
    ComPtr<IBaseFilter> filter;
    filter.Attach(info.pFilter);
    log << "Downstream pin: QueryPinInfo=" << Hex(result) << " address=" << pin;
    if (result == S_OK) {
        log << " name=" << Utf8(info.achName) << " direction=" << info.dir;
    }
    log << std::endl;

    if (filter) {
        FILTER_INFO filterInfo {};
        const HRESULT infoResult = filter->QueryFilterInfo(&filterInfo);
        ComPtr<IFilterGraph> graph;
        graph.Attach(filterInfo.pGraph);
        CLSID clsid {};
        const HRESULT clsidResult = filter->GetClassID(&clsid);

        log << "Downstream filter: name=" << (infoResult == S_OK ? Utf8(filterInfo.achName) : "<unavailable>")
            << " CLSID=" << (clsidResult == S_OK ? GuidText(clsid) : Hex(clsidResult)) << std::endl;
    }
}

struct Candidate {
    const char *name;
    GUID subtype;
    DWORD compression;
    WORD storageBits;
    unsigned chromaDivisor; // 2 = 4:2:2, 4 = 4:2:0, 1 = full resolution / RGB
    bool rgb = false;
    bool dib = false;
    DWORD redMask = 0;
    DWORD greenMask = 0;
    DWORD blueMask = 0;
};

auto FourccCandidate(const char *name, const char *fourcc, WORD storageBits,
                     unsigned chromaDivisor, bool rgb = false) -> Candidate {
    const DWORD compression = Fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]);

    return {name, FOURCCMap(compression), compression, storageBits, chromaDivisor, rgb};
}

auto BuildCandidate(const Candidate &candidate, const CMediaType &reference,
                    bool videoInfo2, bool tagged, bool topDown, CMediaType &type) -> HRESULT {
    const auto *source = VideoHeader(reference);
    if (source == nullptr || source->biWidth <= 0 || source->biHeight == 0) {
        return E_INVALIDARG;
    }

    const int64_t width = source->biWidth;
    const int64_t height = source->biHeight < 0 ? -int64_t(source->biHeight) : source->biHeight;
    if ((candidate.chromaDivisor >= 2 && width % 2 != 0) ||
        (candidate.chromaDivisor == 4 && height % 2 != 0)) {
        return E_INVALIDARG;
    }

    const uint64_t rowBytes = candidate.dib ? ((width * candidate.storageBits + 31) / 32) * 4
                                          : (width * candidate.storageBits) / 8;
    if (height > (std::numeric_limits<LONG>::max)() ||
        rowBytes > static_cast<uint64_t>((std::numeric_limits<LONG>::max)()) / height) {
        return E_INVALIDARG;
    }
    const uint64_t imageBytes = rowBytes * height;

    type.InitMediaType();
    type.SetType(&MEDIATYPE_Video);
    type.SetSubtype(&candidate.subtype);
    type.SetFormatType(videoInfo2 ? &FORMAT_VideoInfo2 : &FORMAT_VideoInfo);
    type.SetTemporalCompression(FALSE);
    type.SetSampleSize(static_cast<ULONG>(imageBytes));

    const ULONG headerBytes = videoInfo2 ? sizeof(VIDEOINFOHEADER2) : sizeof(VIDEOINFOHEADER);
    const ULONG maskBytes = candidate.compression == BI_BITFIELDS ? 3 * sizeof(DWORD) : 0;
    const ULONG formatBytes = headerBytes + maskBytes;
    BYTE *buffer = type.AllocFormatBuffer(formatBytes);
    if (buffer == nullptr) {
        return E_OUTOFMEMORY;
    }
    ZeroMemory(buffer, formatBytes);

    REFERENCE_TIME frameTime = 0;
    DWORD aspectX = static_cast<DWORD>(width);
    DWORD aspectY = static_cast<DWORD>(height);
    DWORD sourceFlags = 0;
    if (reference.formattype == FORMAT_VideoInfo2) {
        const auto *header = reinterpret_cast<const VIDEOINFOHEADER2 *>(reference.pbFormat);
        frameTime = header->AvgTimePerFrame;
        aspectX = header->dwPictAspectRatioX != 0 ? header->dwPictAspectRatioX : aspectX;
        aspectY = header->dwPictAspectRatioY != 0 ? header->dwPictAspectRatioY : aspectY;
        sourceFlags = header->dwControlFlags;
    } else {
        frameTime = reinterpret_cast<const VIDEOINFOHEADER *>(reference.pbFormat)->AvgTimePerFrame;
    }

    BITMAPINFOHEADER *bih = nullptr;
    const RECT rect {0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    if (videoInfo2) {
        auto *header = reinterpret_cast<VIDEOINFOHEADER2 *>(buffer);
        header->rcSource = header->rcTarget = rect;
        header->AvgTimePerFrame = frameTime;
        header->dwPictAspectRatioX = aspectX;
        header->dwPictAspectRatioY = aspectY;
        bih = &header->bmiHeader;

        if (tagged) {
            DXVA_ExtendedFormat color {};
            if ((sourceFlags & AMCONTROL_USED) && (sourceFlags & AMCONTROL_COLORINFO_PRESENT)) {
                color = std::bit_cast<DXVA_ExtendedFormat>(sourceFlags);
            }

            color.SampleFormat = AMCONTROL_USED | AMCONTROL_COLORINFO_PRESENT;
            color.NominalRange = candidate.rgb ? DXVA_NominalRange_0_255 : DXVA_NominalRange_16_235;
            color.VideoTransferMatrix = candidate.rgb ? DXVA_VideoTransferMatrix_Unknown : DXVA_VideoTransferMatrix_BT709;
            header->dwControlFlags = std::bit_cast<DWORD>(color);
        }
    } else {
        auto *header = reinterpret_cast<VIDEOINFOHEADER *>(buffer);
        header->rcSource = header->rcTarget = rect;
        header->AvgTimePerFrame = frameTime;
        bih = &header->bmiHeader;
    }

    bih->biSize = sizeof(BITMAPINFOHEADER);
    bih->biWidth = static_cast<LONG>(width);
    bih->biHeight = static_cast<LONG>(topDown ? -height : height);
    bih->biPlanes = 1;
    bih->biBitCount = candidate.storageBits;
    bih->biCompression = candidate.compression;
    bih->biSizeImage = static_cast<DWORD>(imageBytes);

    if (candidate.compression == BI_BITFIELDS) {
        // BITMAPINFOHEADER is followed by three DWORD masks in R, G, B order.
        // The two alpha bits are identified by the DirectShow subtype.
        auto *masks = reinterpret_cast<DWORD *>(bih + 1);
        masks[0] = candidate.redMask;
        masks[1] = candidate.greenMask;
        masks[2] = candidate.blueMask;
    }

    return S_OK;
}

auto LogQuery(std::ostream &log, IPin *pin, const AM_MEDIA_TYPE &type) -> HRESULT {
    // Flush before calling third-party code so even a partial report identifies the last probe.
    log << "  " << DescribeMediaType(type) << std::endl;
    const HRESULT result = pin->QueryAccept(&type);
    log << "  QueryAccept=" << (result == S_OK ? "ACCEPTED" : result == S_FALSE ? "REJECTED" : "ERROR")
        << " (" << Hex(result) << ")" << std::endl;

    return result;
}

auto EnumeratePreferredTypes(std::ostream &log, IPin *pin) -> void {
    log << "\nAdvertised preferred types (may be incomplete or empty):" << std::endl;
    ComPtr<IEnumMediaTypes> types;
    const HRESULT result = pin->EnumMediaTypes(types.GetAddressOf());
    log << "EnumMediaTypes=" << Hex(result) << std::endl;
    if (result != S_OK || !types) {
        return;
    }

    // Protect graph construction from a broken enumerator that never terminates.
    for (unsigned index = 0; index < 256; ++index) {
        AM_MEDIA_TYPE *type = nullptr;
        const HRESULT next = types->Next(1, &type, nullptr);
        if (next != S_OK || type == nullptr) {
            if (type != nullptr) { DeleteMediaType(type); }
            log << "Enumeration ended after " << index << " entries: " << Hex(next) << std::endl;
            return;
        }

        // Own the COM allocation even if formatting or the query throws.
        const auto deleter = [](AM_MEDIA_TYPE *value) { DeleteMediaType(value); };
        std::unique_ptr<AM_MEDIA_TYPE, decltype(deleter)> owned(type, deleter);
        // Preference entries may be partial types. Do not pass an incomplete type to QueryAccept.
        log << "Advertised[" << index << "]: " << DescribeMediaType(*type) << std::endl;
    }

    log << "Enumeration truncated at 256 entries." << std::endl;
}

}

auto LogDownstreamFormats(IPin *pin, const CMediaType *reference) noexcept -> void {
    try {
        auto log = OpenLog();
        if (!log || pin == nullptr) {
            OutputDebugStringW(L"MLFilter: downstream format diagnostic log could not be opened.\n");
            return;
        }

        SYSTEMTIME now {};
        GetLocalTime(&now);
        log << "\n=== MLFilter downstream format diagnostic " << now.wYear << '-' << now.wMonth << '-'
            << now.wDay << ' ' << now.wHour << ':' << now.wMinute << ':' << now.wSecond
            << " PID=" << GetCurrentProcessId() << " TID=" << GetCurrentThreadId() << " ===" << std::endl;
        LogIdentity(log, pin);
        log << "Only the immediate downstream pin is probed; it may be a transform, not the renderer.\n"
            << "ACCEPTED means QueryAccept returned S_OK. It does not prove connection, playback,\n"
            << "GPU transport, HDR handling, or preservation of the format's bit depth.\n"
            << "Only the listed media types and current output dimensions are checked.\n" << std::endl;

        EnumeratePreferredTypes(log, pin);
        if (reference == nullptr || VideoHeader(*reference) == nullptr) {
            log << "Candidate probes skipped: MLFilter output media type unavailable." << std::endl;
            return;
        }

        log << "\nCurrent MLFilter output type:" << std::endl;
        LogQuery(log, pin, *reference);

        const Candidate candidates[] = {
            FourccCandidate("RGB48 (RGB0)", "RGB0", 48, 1, true),
            FourccCandidate("BGR48 (BGR0)", "BGR0", 48, 1, true),
            FourccCandidate("Y416", "Y416", 64, 1),
            FourccCandidate("Y410", "Y410", 32, 1),
            // SDK DirectShow GUIDs; these are distinct from the Media Foundation RGB GUIDs.
            // Probe both subtype-only BI_RGB and explicit BI_BITFIELDS representations.
            {"A2B10G10R10 (BI_RGB)", MEDIASUBTYPE_A2B10G10R10, BI_RGB, 32, 1, true, true},
            {"A2B10G10R10 (BI_BITFIELDS)", MEDIASUBTYPE_A2B10G10R10, BI_BITFIELDS, 32, 1, true, true,
             0x000003FF, 0x000FFC00, 0x3FF00000},
            {"A2R10G10B10 (BI_RGB)", MEDIASUBTYPE_A2R10G10B10, BI_RGB, 32, 1, true, true},
            {"A2R10G10B10 (BI_BITFIELDS)", MEDIASUBTYPE_A2R10G10B10, BI_BITFIELDS, 32, 1, true, true,
             0x3FF00000, 0x000FFC00, 0x000003FF},
            FourccCandidate("P216", "P216", 32, 2),
            FourccCandidate("P210", "P210", 32, 2),
            FourccCandidate("Y216", "Y216", 32, 2),
            FourccCandidate("Y210", "Y210", 32, 2),
            FourccCandidate("P016", "P016", 24, 4),
            FourccCandidate("P010", "P010", 24, 4),
            {"RGB32", MEDIASUBTYPE_RGB32, BI_RGB, 32, 1, true, true},
            {"ARGB32", MEDIASUBTYPE_ARGB32, BI_RGB, 32, 1, true, true},
            {"RGB24", MEDIASUBTYPE_RGB24, BI_RGB, 24, 1, true, true},
            FourccCandidate("AYUV", "AYUV", 32, 1),
            FourccCandidate("YV24", "YV24", 24, 1),
            FourccCandidate("YV16", "YV16", 16, 2),
            FourccCandidate("YUY2", "YUY2", 16, 2),
            FourccCandidate("UYVY", "UYVY", 16, 2),
            FourccCandidate("NV12", "NV12", 12, 4),
            FourccCandidate("YV12", "YV12", 12, 4),
            FourccCandidate("I420", "I420", 12, 4),
            FourccCandidate("IYUV", "IYUV", 12, 4),
        };

        log << "\nCandidate probes: VideoInfo2 without/with color tags, and VideoInfo without tags.\n"
            << "RGB DIBs are checked with positive and negative heights; FOURCC types use positive heights.\n"
            << "YUV tagged variants use limited-range BT.709; untagged variants leave color unspecified."
            << std::endl;
        for (const auto &candidate : candidates) {
            bool accepted = false;
            for (unsigned variant = 0; variant < (candidate.dib ? 6U : 3U); ++variant) {
                const bool topDown = variant >= 3;
                const bool videoInfo2 = variant % 3 != 2;
                const bool tagged = variant % 3 == 1;
                log << "\nProbe " << candidate.name << " " << (videoInfo2 ? "VideoInfo2" : "VideoInfo")
                    << (tagged ? " tagged" : " untagged") << (topDown ? " negative-height" : " positive-height")
                    << std::endl;

                CMediaType type;
                const HRESULT build = BuildCandidate(candidate, *reference, videoInfo2, tagged, topDown, type);
                if (build != S_OK) {
                    log << "  SKIPPED: invalid dimensions/layout or allocation failure (" << Hex(build) << ")"
                        << std::endl;
                    continue;
                }

                const HRESULT result = LogQuery(log, pin, type);
                accepted = accepted || result == S_OK;
            }

            log << "SUMMARY " << candidate.name << ": " << (accepted ? "ACCEPTED in at least one variant"
                                                                        : "NO tested variant accepted") << std::endl;
        }

        log << "\n=== Format probes complete; normal RGB48 negotiation follows ===" << std::endl;
    } catch (...) {
        OutputDebugStringW(L"MLFilter: downstream format diagnostics failed; normal connection will continue.\n");
    }
}

auto LogDownstreamConnectionResult(HRESULT result, const CMediaType *connected) noexcept -> void {
    try {
        auto log = OpenLog();
        if (log) {
            log << "Normal MLFilter connection result (TID=" << GetCurrentThreadId() << "): " << Hex(result)
                << std::endl;
            if (connected != nullptr) {
                log << "Connected type: " << DescribeMediaType(*connected) << std::endl;
            }
        }
    } catch (...) {
        OutputDebugStringW(L"MLFilter: could not log downstream connection result.\n");
    }
}

}
