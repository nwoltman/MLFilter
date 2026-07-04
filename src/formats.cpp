// SPDX-License-Identifier: Apache-2.0

#include "formats.h"

#include <streams.h>

namespace MLFilter {

auto SupportedInputSubtypes() -> const std::vector<GUID> & {
    static const std::vector<GUID> subtypes = [] {
        // YUV formats are FOURCC-based subtypes.
        static const DWORD yuvFourccs[] = {
            // 4:2:0
            Fourcc('N', 'V', '1', '2'), // 8-bit
            Fourcc('P', '0', '1', '0'), // 10-bit
        };

        std::vector<GUID> result;
        result.reserve(std::size(yuvFourccs));
        for (const DWORD fourcc : yuvFourccs) {
            result.push_back(FOURCCMap(fourcc));
        }
        return result;
    }();

    return subtypes;
}

auto IsSupportedInputSubtype(const GUID &subtype) -> bool {
    for (const GUID &candidate : SupportedInputSubtypes()) {
        if (subtype == candidate) {
            return true;
        }
    }
    return false;
}

auto OutputSubtype() -> const GUID & {
    // RGB48: 16-bit-per-channel packed RGB, FOURCC 'RGB0' (same subtype LAV/avisynth_filter use).
    static const GUID subtype = FOURCCMap(Fourcc('R', 'G', 'B', '0'));
    return subtype;
}

}
