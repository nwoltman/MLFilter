// SPDX-License-Identifier: Apache-2.0

#include "output_pin.h"

#include "filter.h"
#include "formats.h"

namespace MLFilter {

MLFilterOutputPin::MLFilterOutputPin(CMLFilter *filter, HRESULT *result)
    : CTransformOutputPin(NAME("Transform output pin"), filter, result, L"XForm Out")
    , _filter(filter) {}

auto STDMETHODCALLTYPE MLFilterOutputPin::Connect(IPin *receivePin, const AM_MEDIA_TYPE *mediaType) -> HRESULT {
    CAutoLock lock(m_pLock);

    // An explicit subtype or format block constrains the connection to the caller's request.
    if (mediaType != nullptr &&
        ((mediaType->majortype != GUID_NULL && mediaType->majortype != MEDIATYPE_Video) ||
         mediaType->subtype != GUID_NULL || mediaType->formattype != GUID_NULL)) {
        return CTransformOutputPin::Connect(receivePin, mediaType);
    }

    // Let DirectShow try both pins' RGB48 proposals before it considers Y410.
    CMediaType preferred;
    preferred.SetType(&MEDIATYPE_Video);
    preferred.SetSubtype(&Rgb48Subtype());
    const HRESULT hr = CTransformOutputPin::Connect(receivePin, &preferred);
    if (!ShouldTryY410(hr)) {
        return hr;
    }

    preferred.SetSubtype(&Y410Subtype());
    const HRESULT fallbackHr = CTransformOutputPin::Connect(receivePin, &preferred);
    if (fallbackHr == S_OK) {
        OutputDebugStringW(L"MLFilter: downstream accepted Y410 after RGB48 negotiation failed.\n");
    }

    return fallbackHr;
}

auto MLFilterOutputPin::ShouldTryY410(HRESULT rgb48Result) -> bool {
    if (rgb48Result != VFW_E_NO_ACCEPTABLE_TYPES && rgb48Result != VFW_E_TYPE_NOT_ACCEPTED &&
        rgb48Result != VFW_E_INVALIDMEDIATYPE) {
        return false;
    }

    // The base class can return NO_ACCEPTABLE_TYPES even if we had no type to offer.
    CMediaType rgb48;
    return GetMediaType(0, &rgb48) == S_OK;
}

auto MLFilterOutputPin::Inactive() -> HRESULT {
    _filter->ReleaseOutputRegistrations();
    return CTransformOutputPin::Inactive();
}

}
