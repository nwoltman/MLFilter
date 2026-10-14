// SPDX-License-Identifier: Apache-2.0

#include "output_pin.h"

#include "filter.h"
#include "renderer_diagnostics.h"

namespace MLFilter {

MLFilterOutputPin::MLFilterOutputPin(CMLFilter *filter, HRESULT *result)
    : CTransformOutputPin(NAME("Transform output pin"), filter, result, L"XForm Out")
    , _filter(filter) {}

auto STDMETHODCALLTYPE MLFilterOutputPin::Connect(IPin *receivePin, const AM_MEDIA_TYPE *mediaType) -> HRESULT {
    CAutoLock lock(m_pLock);

    // TEMPORARY: probe before negotiation, including when RGB48 cannot connect to the renderer.
    // Remove this block and the result logging below after collecting the PotPlayer report.
    if (receivePin != nullptr && !IsConnected() && IsStopped()) {
        CMediaType reference;
        const HRESULT referenceResult = _filter->GetMediaType(0, &reference);
        LogDownstreamFormats(receivePin, referenceResult == S_OK ? &reference : nullptr);
    }

    const HRESULT result = CTransformOutputPin::Connect(receivePin, mediaType);
    LogDownstreamConnectionResult(result, result == S_OK ? &CurrentMediaType() : nullptr);

    return result;
}

auto MLFilterOutputPin::Inactive() -> HRESULT {
    _filter->ReleaseOutputRegistrations();
    return CTransformOutputPin::Inactive();
}

}
