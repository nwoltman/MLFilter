// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <windows.h>

namespace MLFilter {

// The decoded video subtypes the filter accepts on its input pin. First cut: 4:2:0 NV12
// (8-bit) and P010 (10-bit), which cover the overwhelming majority of decoded content.
auto SupportedInputSubtypes() -> const std::vector<GUID> &;

auto IsSupportedInputSubtype(const GUID &subtype) -> bool;

auto Rgb48Subtype() -> const GUID &;
auto Y410Subtype() -> const GUID &;

// Packs four characters into a little-endian FOURCC DWORD (e.g. for FOURCCMap).
inline constexpr auto Fourcc(char a, char b, char c, char d) -> DWORD {
    return static_cast<DWORD>(static_cast<unsigned char>(a))
         | (static_cast<DWORD>(static_cast<unsigned char>(b)) << 8)
         | (static_cast<DWORD>(static_cast<unsigned char>(c)) << 16)
         | (static_cast<DWORD>(static_cast<unsigned char>(d)) << 24);
}

inline constexpr DWORD kRgb48Fourcc = Fourcc('R', 'G', 'B', '0');
inline constexpr DWORD kY410Fourcc = Fourcc('Y', '4', '1', '0');

}
