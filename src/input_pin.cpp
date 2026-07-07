// SPDX-License-Identifier: Apache-2.0

#include "input_pin.h"

#include "filter.h"
#include "formats.h"

#include <cuda_d3d11_interop.h>
#include <dxgi.h>

#include <optional>

namespace MLFilter {
namespace {

auto CudaDxgiAdapterIndex() -> std::optional<UINT> {
    int currentCudaDevice = 0;
    if (cudaGetDevice(&currentCudaDevice) != cudaSuccess) {
        cudaGetLastError();
        return std::nullopt;
    }

    IDXGIFactory *factory = nullptr;
    if (FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory), reinterpret_cast<void **>(&factory)))) {
        return std::nullopt;
    }

    std::optional<UINT> result;
    for (UINT index = 0;; ++index) {
        IDXGIAdapter *adapter = nullptr;
        const HRESULT enumResult = factory->EnumAdapters(index, &adapter);
        if (enumResult == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        if (FAILED(enumResult)) {
            result.reset();
            break;
        }

        int adapterCudaDevice = -1;
        const cudaError_t cudaResult = cudaD3D11GetDevice(&adapterCudaDevice, adapter);
        adapter->Release();

        if (cudaResult == cudaSuccess && adapterCudaDevice == currentCudaDevice) {
            result = index;
            break;
        }

        if (cudaResult != cudaSuccess) {
            cudaGetLastError();
        }
    }

    factory->Release();
    return result;
}

}

MLFilterInputPin::MLFilterInputPin(CMLFilter *filter, HRESULT *phr)
    : CTransformInputPin(NAME("MLFilter input pin"), filter, phr, L"XForm In") {}

MLFilterInputPin::~MLFilterInputPin() {
    ReleaseD3D11State();
}

auto STDMETHODCALLTYPE MLFilterInputPin::QueryInterface(REFIID riid, void **ppv) -> HRESULT {
    return NonDelegatingQueryInterface(riid, ppv);
}

auto STDMETHODCALLTYPE MLFilterInputPin::AddRef() -> ULONG {
    return NonDelegatingAddRef();
}

auto STDMETHODCALLTYPE MLFilterInputPin::Release() -> ULONG {
    return NonDelegatingRelease();
}

auto STDMETHODCALLTYPE MLFilterInputPin::NonDelegatingQueryInterface(REFIID riid, void **ppv) -> HRESULT {
    CheckPointer(ppv, E_POINTER);

    if (riid == __uuidof(ID3D11DecoderConfiguration)) {
        const std::optional<UINT> adapterIndex = CudaDxgiAdapterIndex();
        if (!adapterIndex.has_value()) {
            *ppv = nullptr;
            return E_NOINTERFACE;
        }

        {
            CAutoLock lock(&_d3d11Lock);
            _d3d11AdapterIndex = *adapterIndex;
        }

        return GetInterface(static_cast<ID3D11DecoderConfiguration *>(this), ppv);
    }

    return CTransformInputPin::NonDelegatingQueryInterface(riid, ppv);
}

auto STDMETHODCALLTYPE MLFilterInputPin::ActivateD3D11Decoding(ID3D11Device *device,
                                                               ID3D11DeviceContext *context,
                                                               HANDLE mutex,
                                                               UINT /*flags*/) -> HRESULT {
    CheckPointer(device, E_POINTER);
    CheckPointer(context, E_POINTER);

    CAutoLock lock(&_d3d11Lock);

    if (!IsSupportedInputSubtype(*CurrentMediaType().Subtype())) {
        ReleaseD3D11State();
        return VFW_E_TYPE_NOT_ACCEPTED;
    }

    ReleaseD3D11State();

    _d3d11Device = device;
    _d3d11Device->AddRef();
    _d3d11Context = context;
    _d3d11Context->AddRef();
    _d3d11Mutex = mutex;

    return S_OK;
}

auto STDMETHODCALLTYPE MLFilterInputPin::GetD3D11AdapterIndex() -> UINT {
    CAutoLock lock(&_d3d11Lock);
    return _d3d11AdapterIndex;
}

auto STDMETHODCALLTYPE MLFilterInputPin::GetAllocator(IMemAllocator **allocator) -> HRESULT {
    CheckPointer(allocator, E_POINTER);
    *allocator = nullptr;

    if (IsConnected() && IsSupportedInputSubtype(*CurrentMediaType().Subtype())) {
        // Let the upstream output pin use its own allocator. LAV's D3D11 native path only creates
        // IMediaSampleD3D11 samples from its output-pin allocator; returning our default memory
        // allocator here forces host-backed samples even after ActivateD3D11Decoding succeeds.
        return E_NOTIMPL;
    }

    return CTransformInputPin::GetAllocator(allocator);
}

auto MLFilterInputPin::D3D11State() const -> D3D11DecoderState {
    CAutoLock lock(const_cast<CCritSec *>(&_d3d11Lock));

    D3D11DecoderState state {
        .device = _d3d11Device,
        .context = _d3d11Context,
        .mutex = _d3d11Mutex,
    };

    if (state.device != nullptr) {
        state.device->AddRef();
    }
    if (state.context != nullptr) {
        state.context->AddRef();
    }

    return state;
}

auto MLFilterInputPin::BreakConnect() -> HRESULT {
    {
        CAutoLock lock(&_d3d11Lock);
        ReleaseD3D11State();
    }

    return CTransformInputPin::BreakConnect();
}

auto MLFilterInputPin::ReleaseD3D11State() -> void {
    if (_d3d11Context != nullptr) {
        _d3d11Context->Release();
        _d3d11Context = nullptr;
    }
    if (_d3d11Device != nullptr) {
        _d3d11Device->Release();
        _d3d11Device = nullptr;
    }

    _d3d11Mutex = nullptr;
}

}
