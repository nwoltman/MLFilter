// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <windows.h>

namespace MLFilter {

// The decoded/uncompressed video subtypes the filter accepts. This is the single source
// of truth used both for filter registration and for CheckInputType at connection time.
auto SupportedSubtypes() -> const std::vector<GUID> &;

auto IsSupportedSubtype(const GUID &subtype) -> bool;

}
