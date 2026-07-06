// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <streams.h>

namespace MLFilter {

class CMLFilter;

// Releases cached CUDA registrations before the downstream allocator decommits its buffers.
class MLFilterOutputPin final : public CTransformOutputPin {
public:
    MLFilterOutputPin(CMLFilter *filter, HRESULT *result);

    auto Inactive() -> HRESULT override;

private:
    CMLFilter *_filter;
};

}
