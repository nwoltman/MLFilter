// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <windows.h>

namespace MLFilter {

// The decoded video subtypes the filter accepts on its input pin. First cut: 4:2:0 NV12
// (8-bit) and P010 (10-bit), which cover the overwhelming majority of decoded content.
auto SupportedInputSubtypes() -> const std::vector<GUID> &;

auto IsSupportedInputSubtype(const GUID &subtype) -> bool;

// The single subtype the filter emits downstream: 16-bit-per-channel packed RGB (RGB48).
auto OutputSubtype() -> const GUID &;

// Byte stride of one RGB48 row (6 bytes per pixel). LAV/madVR use an unpadded width*6 stride
// for this packed FOURCC format. Used identically by the output media type, allocator sizing,
// and the RGB48 packing so they agree.
inline auto Rgb48Stride(int width) -> int { return width * 6; }

// Packs four characters into a little-endian FOURCC DWORD (e.g. for FOURCCMap).
inline constexpr auto Fourcc(char a, char b, char c, char d) -> DWORD {
    return static_cast<DWORD>(static_cast<unsigned char>(a))
         | (static_cast<DWORD>(static_cast<unsigned char>(b)) << 8)
         | (static_cast<DWORD>(static_cast<unsigned char>(c)) << 16)
         | (static_cast<DWORD>(static_cast<unsigned char>(d)) << 24);
}

}
