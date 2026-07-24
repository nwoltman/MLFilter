// SPDX-License-Identifier: Apache-2.0

#pragma once

#define MLFILTER_STRINGIFY_IMPL(value) #value
#define MLFILTER_STRINGIFY(value) MLFILTER_STRINGIFY_IMPL(value)
#define MLFILTER_WIDEN_IMPL(value) L##value
#define MLFILTER_WIDEN(value) MLFILTER_WIDEN_IMPL(value)

namespace MLFilter {

inline constexpr wchar_t VERSION_LABEL[] =
    L"MLFilter " MLFILTER_WIDEN(MLFILTER_STRINGIFY(MLFILTER_VERSION));

}
