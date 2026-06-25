// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <string>

#include <streams.h>

namespace MLFilter {

// Phase 1 filter: a pass-through transform-in-place filter that delivers frames
// downstream unmodified, and exposes a configuration property page. Inference
// will be added in a later phase.
class CMLFilter
    : public CTransInPlaceFilter
    , public ISpecifyPropertyPages {
public:
    CMLFilter(LPUNKNOWN pUnk, HRESULT *phr);

    static auto CALLBACK CreateInstance(LPUNKNOWN pUnk, HRESULT *phr) -> CUnknown *;

    DECLARE_IUNKNOWN
    auto STDMETHODCALLTYPE NonDelegatingQueryInterface(REFIID riid, void **ppv) -> HRESULT override;

    // CTransInPlaceFilter
    auto CheckInputType(const CMediaType *mtIn) -> HRESULT override;
    auto CompleteConnect(PIN_DIRECTION direction, IPin *pReceivePin) -> HRESULT override;
    auto Transform(IMediaSample *pSample) -> HRESULT override;

    // ISpecifyPropertyPages
    auto STDMETHODCALLTYPE GetPages(CAUUID *pPages) -> HRESULT override;

private:
    // Builds (if not already cached) the engine matching the connected input's
    // resolution, showing a progress window during the build.
    auto EnsureEngineForInput() -> void;

    // True if the currently playing file should be processed: either no glob filter is
    // configured, the source file can't be determined, or the file matches a glob.
    auto ShouldProcessCurrentFile() -> bool;

    // Walks the graph for an IFileSourceFilter and returns the file path being played
    // (empty if none/unknown).
    auto GetSourceFilePath() const -> std::wstring;

    // Removes this filter from the graph on a worker thread (deferred to avoid
    // re-entrancy while the graph is being built).
    auto ScheduleSelfRemoval() -> void;

    std::atomic<bool> _removalScheduled = false;
};

}
