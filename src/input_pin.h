// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <streams.h>

#include "d3d11_native_interfaces.h"

namespace MLFilter {

class CMLFilter;

// Releases cached CUDA registrations before the upstream allocator decommits its buffers.
class MLFilterInputPin
    : public CTransformInputPin
    , public ID3D11DecoderConfiguration {
public:
    MLFilterInputPin(CMLFilter *filter, HRESULT *phr);
    ~MLFilterInputPin() override;

    auto STDMETHODCALLTYPE QueryInterface(REFIID riid, void **ppv) -> HRESULT override;
    auto STDMETHODCALLTYPE AddRef() -> ULONG override;
    auto STDMETHODCALLTYPE Release() -> ULONG override;
    auto STDMETHODCALLTYPE NonDelegatingQueryInterface(REFIID riid, void **ppv) -> HRESULT override;

    auto STDMETHODCALLTYPE ActivateD3D11Decoding(ID3D11Device *device,
                                                 ID3D11DeviceContext *context,
                                                 HANDLE mutex,
                                                 UINT flags) -> HRESULT override;
    auto STDMETHODCALLTYPE GetD3D11AdapterIndex() -> UINT override;
    auto STDMETHODCALLTYPE GetAllocator(IMemAllocator **allocator) -> HRESULT override;

    auto D3D11State() const -> D3D11DecoderState;
    auto Inactive() -> HRESULT override;
    auto BreakConnect() -> HRESULT;

private:
    auto ReleaseD3D11State() -> void;

    CCritSec _d3d11Lock;
    ID3D11Device *_d3d11Device = nullptr;
    ID3D11DeviceContext *_d3d11Context = nullptr;
    HANDLE _d3d11Mutex = nullptr;
    UINT _d3d11AdapterIndex = 0;
    CMLFilter *_filter;
};

}
