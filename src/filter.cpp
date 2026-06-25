// SPDX-License-Identifier: Apache-2.0

#include "filter.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <thread>
#include <vector>

#include <dvdmedia.h>
#include <shlwapi.h>

#include "constants.h"
#include "formats.h"
#include "guids.h"
#include "progress_window.h"
#include "settings.h"
#include "engine/engine_builder.h"
#include "engine/tensorrt_engine_builder.h"

#pragma comment(lib, "shlwapi")

namespace MLFilter {

namespace {

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

// Returns true if the file should be processed: no patterns configured (no
// restriction), unknown file path, or the path matches at least one pattern.
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

// Extracts the pixel width/height from a video media type (VIDEOINFOHEADER or
// VIDEOINFOHEADER2). Returns false if the type isn't a recognized video format.
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

}


CMLFilter::CMLFilter(LPUNKNOWN pUnk, HRESULT *phr)
    : CTransInPlaceFilter(FILTER_NAME_WIDE, pUnk, CLSID_MLFilter, phr, false) {}

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

    return CTransInPlaceFilter::NonDelegatingQueryInterface(riid, ppv);
}

auto CMLFilter::CheckInputType(const CMediaType *mtIn) -> HRESULT {
    CheckPointer(mtIn, E_POINTER);

    if (*mtIn->Type() != MEDIATYPE_Video || !IsSupportedSubtype(*mtIn->Subtype())) {
        // Reject compressed/unknown types so we connect after the decoder, not before it.
        return VFW_E_TYPE_NOT_ACCEPTED;
    }

    // If the playing file doesn't match the configured globs, refuse the connection so
    // the graph routes around us, and remove this filter from the graph.
    if (!ShouldProcessCurrentFile()) {
        ScheduleSelfRemoval();
        return VFW_E_TYPE_NOT_ACCEPTED;
    }

    return S_OK;
}

auto CMLFilter::CompleteConnect(PIN_DIRECTION direction, IPin *pReceivePin) -> HRESULT {
    const HRESULT hr = CTransInPlaceFilter::CompleteConnect(direction, pReceivePin);
    if (FAILED(hr)) {
        return hr;
    }

    // The input's media type (and thus the real video resolution) is known once the
    // upstream pin connects, before playback starts.
    if (direction == PINDIR_INPUT) {
        EnsureEngineForInput();
    }

    return hr;
}

auto CMLFilter::EnsureEngineForInput() -> void {
    Settings settings;
    settings.Load();
    if (settings.modelPath.empty() || !std::filesystem::exists(settings.modelPath)) {
        return;  // no model configured: pure pass-through
    }

    int width = 0;
    int height = 0;
    if (m_pInput == nullptr || !GetVideoResolution(m_pInput->CurrentMediaType(), width, height)) {
        return;
    }

    TensorRTEngineBuilder builder;
    const EngineBuildRequest request { .onnxPath = settings.modelPath, .width = width, .height = height };
    if (builder.Exists(request)) {
        return;  // already built for this resolution/GPU/driver
    }

    // Build synchronously (blocks graph completion until ready) with a responsive
    // progress window on its own thread.
    ProgressWindow window;
    window.Open(FILTER_NAME_WIDE);

    const EngineBuildResult result = builder.Build(request, [&window](const std::wstring &message) {
        window.Log(message);
    });

    if (result.success) {
        window.Log(L"Engine ready. Starting playback...");
    } else {
        window.Log(L"Engine build failed; the video will play without processing.");
        window.Log(result.message);
    }

    // Let the final status be visible briefly before playback proceeds.
    Sleep(result.success ? 600 : 2500);
    window.Close();
}

auto CMLFilter::ShouldProcessCurrentFile() -> bool {
    Settings settings;
    settings.Load();
    if (Trim(settings.fileGlobs).empty()) {
        return true;  // no glob filter configured
    }
    return FileMatchesGlobs(GetSourceFilePath(), settings.fileGlobs);
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

    // Keep both objects alive for the duration of the asynchronous removal. The removal
    // runs on its own thread so it doesn't re-enter the graph while it is being built;
    // RemoveFilter blocks until the graph lock is free, then detaches this filter.
    graph->AddRef();
    AddRef();
    std::thread([this, graph]() {
        graph->RemoveFilter(this);
        graph->Release();
        Release();
    }).detach();
}

auto CMLFilter::Transform(IMediaSample * /*pSample*/) -> HRESULT {
    // Pass-through: the sample is delivered downstream unchanged.
    return S_OK;
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
