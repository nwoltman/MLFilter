// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <memory>
#include <string>

#include <streams.h>

#include "frame_processor.h"

namespace MLFilter {

// Copy transform: converts a supported 4:2:0 YUV input frame to RGB, runs the TensorRT engine, and
// emits RGB48 downstream (a different pixel format and possibly a larger resolution than the
// input, which is why this is a copy transform rather than transform-in-place).
class CMLFilter
    : public CTransformFilter
    , public ISpecifyPropertyPages {
public:
    CMLFilter(LPUNKNOWN pUnk, HRESULT *phr);

    static auto CALLBACK CreateInstance(LPUNKNOWN pUnk, HRESULT *phr) -> CUnknown *;

    DECLARE_IUNKNOWN
    auto STDMETHODCALLTYPE NonDelegatingQueryInterface(REFIID riid, void **ppv) -> HRESULT override;

    // CTransformFilter
    auto CheckInputType(const CMediaType *mtIn) -> HRESULT override;
    auto CheckTransform(const CMediaType *mtIn, const CMediaType *mtOut) -> HRESULT override;
    auto GetMediaType(int iPosition, CMediaType *pMediaType) -> HRESULT override;
    auto DecideBufferSize(IMemAllocator *pAlloc, ALLOCATOR_PROPERTIES *pProps) -> HRESULT override;
    auto Transform(IMediaSample *pIn, IMediaSample *pOut) -> HRESULT override;
    auto CompleteConnect(PIN_DIRECTION direction, IPin *pReceivePin) -> HRESULT override;
    auto BreakConnect(PIN_DIRECTION direction) -> HRESULT override;

    // ISpecifyPropertyPages
    auto STDMETHODCALLTYPE GetPages(CAUUID *pPages) -> HRESULT override;

private:
    // Builds (if not already cached) the engine matching the connected input's resolution,
    // showing a progress window during the build.
    auto EnsureEngineForInput() -> void;

    // Deserializes the cached engine and builds the conversion pipeline for the connected input.
    // A null processor means the filter cannot operate and must remove itself from the graph.
    auto SetupProcessor() -> void;

    // True if the model selected for the current resolution is available, the input passes
    // the optional 1080p limit, and the playing file matches the configured glob filter.
    auto ShouldProcessCurrentFile(const CMediaType &inputType) -> bool;

    // Walks the graph for an IFileSourceFilter and returns the file path being played.
    auto GetSourceFilePath() const -> std::wstring;

    // Removes this filter from the graph on a worker thread.
    auto ScheduleSelfRemoval() -> void;

    std::unique_ptr<FrameProcessor> _processor;
    std::atomic<bool> _removalScheduled = false;
    bool _inputFormatSupported = true;
    bool _inputFormatErrorReported = false;
    std::wstring _inputFormatDescription;
};

}
