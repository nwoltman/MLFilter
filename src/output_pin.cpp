// SPDX-License-Identifier: Apache-2.0

#include "output_pin.h"

#include "filter.h"

namespace MLFilter {

MLFilterOutputPin::MLFilterOutputPin(CMLFilter *filter, HRESULT *result)
    : CTransformOutputPin(NAME("Transform output pin"), filter, result, L"XForm Out")
    , _filter(filter) {}

auto MLFilterOutputPin::Inactive() -> HRESULT {
    _filter->ReleaseOutputRegistrations();
    return CTransformOutputPin::Inactive();
}

}
