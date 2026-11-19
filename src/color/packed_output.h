// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace MLFilter {

enum class PackedOutputFormat {
    Rgb48,
    Y410,
};

constexpr auto PackedOutputBytesPerPixel(PackedOutputFormat format) -> int {
    return format == PackedOutputFormat::Y410 ? 4 : 6;
}

// Limited-range 10-bit Y'CbCr codes with neutral chroma and opaque alpha.
inline constexpr unsigned kY410Black = (3u << 30) | (512u << 20) | (64u << 10) | 512u;
inline constexpr unsigned kY410White = (3u << 30) | (512u << 20) | (940u << 10) | 512u;

}
