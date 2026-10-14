// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <streams.h>

namespace MLFilter {

// Temporary, backend-independent DirectShow diagnostics. Writes UTF-8 reports under
// %LOCALAPPDATA%\MLFilter\Logs before connection negotiation, even if RGB48 is rejected.
// Enumeration is a preference list; QueryAccept does not prove playback or retained precision.
auto LogDownstreamFormats(IPin *pin, const CMediaType *reference) noexcept -> void;
auto LogDownstreamConnectionResult(HRESULT result, const CMediaType *connected) noexcept -> void;

}
