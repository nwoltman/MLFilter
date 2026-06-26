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
            Fourcc('Y', 'V', '1', '2'),
            Fourcc('P', '0', '1', '0'), // 10-bit
            Fourcc('P', '0', '1', '6'), // 16-bit
            // 4:2:2
            Fourcc('Y', 'U', 'Y', '2'), // 8-bit
            Fourcc('U', 'Y', 'V', 'Y'),
            Fourcc('P', '2', '1', '0'), // 10-bit
            Fourcc('v', '2', '1', '0'),
            Fourcc('P', '2', '1', '6'), // 16-bit
            // 4:4:4
            Fourcc('Y', 'V', '2', '4'), // 8-bit
            Fourcc('A', 'Y', 'U', 'V'),
            Fourcc('Y', '4', '1', '0'), // 10-bit
            Fourcc('v', '4', '1', '0'),
            Fourcc('Y', '4', '1', '6'), // 16-bit
        };

        std::vector<GUID> result;
        result.reserve(std::size(yuvFourccs) + 3);
        for (const DWORD fourcc : yuvFourccs) {
            result.push_back(FOURCCMap(fourcc));
        }

        // RGB
        result.push_back(MEDIASUBTYPE_RGB32); // 8-bit
        result.push_back(MEDIASUBTYPE_RGB24);
        result.push_back(FOURCCMap(Fourcc('R', 'G', 'B', '0'))); // RGB48, 16-bit

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
