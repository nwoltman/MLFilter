// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace MLFilter {

// Filter configuration persisted under HKCU\Software\MLFilter.
class Settings {
public:
    auto Load() -> void;
    auto Save() const -> void;

    std::wstring modelPath;
    bool onlyRun1080pOrLower = true;

    // Newline-separated wildcard patterns (e.g. "*.mkv"). If non-empty, the filter only
    // processes files whose path matches at least one pattern; otherwise it removes
    // itself from the graph. Empty means "process everything".
    std::wstring fileGlobs;
};

}
