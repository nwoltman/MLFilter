// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <d3d11.h>

namespace MLFilter {

// Minimal ABI-compatible declarations for LAV Filters' D3D11 native transport.
// The UUIDs are the interoperability contract; keep this header intentionally
// small so MLFilter does not depend on LAV's GPL headers.
MIDL_INTERFACE("2BB66002-46B7-4F13-9036-7053328742BE")
ID3D11DecoderConfiguration : public IUnknown {
    virtual auto STDMETHODCALLTYPE ActivateD3D11Decoding(ID3D11Device *device,
                                                         ID3D11DeviceContext *context,
                                                         HANDLE mutex,
                                                         UINT flags) -> HRESULT = 0;
    virtual auto STDMETHODCALLTYPE GetD3D11AdapterIndex() -> UINT = 0;
};

MIDL_INTERFACE("BC8753F5-0AC8-4806-8E5F-A12B2AFE153E")
IMediaSampleD3D11 : public IUnknown {
    virtual auto STDMETHODCALLTYPE GetD3D11Texture(int view,
                                                   ID3D11Texture2D **texture,
                                                   UINT *arraySlice) -> HRESULT = 0;
};

struct D3D11DecoderState {
    ID3D11Device *device = nullptr;
    ID3D11DeviceContext *context = nullptr;
    HANDLE mutex = nullptr;
};

}
