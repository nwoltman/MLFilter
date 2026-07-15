// SPDX-License-Identifier: Apache-2.0
#include "debug_overlay.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cctype>
#include <cstring>
#include <string_view>

namespace MLFilter {
namespace {

struct Glyph { char c; std::array<uint8_t, 7> rows; };
constexpr Glyph GLYPHS[] = {
    {' ',{0,0,0,0,0,0,0}}, {'(',{2,4,8,8,8,4,2}}, {')',{8,4,2,2,2,4,8}},
    {'+',{0,4,4,31,4,4,0}},
    {'-',{0,0,0,31,0,0,0}}, {'.',{0,0,0,0,0,6,6}}, {':',{0,6,6,0,6,6,0}},
    {',',{0,0,0,0,6,4,8}}, {'_',{0,0,0,0,0,0,31}}, {'?',{14,17,1,2,4,0,4}},
    {'/',{1,2,4,8,16,0,0}}, {'0',{14,17,19,21,25,17,14}},
    {'1',{4,12,4,4,4,4,14}}, {'2',{14,17,1,2,4,8,31}},
    {'3',{30,1,1,14,1,1,30}}, {'4',{2,6,10,18,31,2,2}},
    {'5',{31,16,16,30,1,1,30}}, {'6',{14,16,16,30,17,17,14}},
    {'7',{31,1,2,4,8,8,8}}, {'8',{14,17,17,14,17,17,14}},
    {'9',{14,17,17,15,1,1,14}}, {'A',{14,17,17,31,17,17,17}},
    {'B',{30,17,17,30,17,17,30}}, {'C',{14,17,16,16,16,17,14}},
    {'D',{30,17,17,17,17,17,30}},
    {'E',{31,16,16,30,16,16,31}}, {'F',{31,16,16,30,16,16,16}},
    {'G',{14,17,16,23,17,17,15}}, {'I',{14,4,4,4,4,4,14}},
    {'H',{17,17,17,31,17,17,17}}, {'J',{7,2,2,2,2,18,12}},
    {'K',{17,18,20,24,20,18,17}}, {'L',{16,16,16,16,16,16,31}},
    {'M',{17,27,21,21,17,17,17}},
    {'N',{17,25,21,19,17,17,17}}, {'O',{14,17,17,17,17,17,14}},
    {'P',{30,17,17,30,16,16,16}}, {'R',{30,17,17,30,20,18,17}},
    {'Q',{14,17,17,17,21,18,13}},
    {'S',{15,16,16,14,1,1,30}}, {'T',{31,4,4,4,4,4,4}},
    {'U',{17,17,17,17,17,17,14}}, {'V',{17,17,17,17,17,10,4}},
    {'W',{17,17,17,21,21,21,10}}, {'X',{17,17,10,4,10,17,17}},
    {'Y',{17,17,10,4,4,4,4}}, {'Z',{31,1,2,4,8,16,31}},
};

auto FindGlyph(char c) -> const Glyph * {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    for (const auto &glyph : GLYPHS) if (glyph.c == c) return &glyph;
    for (const auto &glyph : GLYPHS) if (glyph.c == '?') return &glyph;
    return &GLYPHS[0]; // unreachable
}

auto PutPixel(uint8_t *frame, size_t stride, int x, int y) -> void {
    auto *p = reinterpret_cast<uint16_t *>(frame + static_cast<size_t>(y) * stride) + x * 3;
    p[0] = p[1] = p[2] = 0xffff;
}

auto DrawText(uint8_t *frame, size_t stride, int width, int height,
              int x, int y, std::string_view text) -> void {
    constexpr int scale = 3;
    for (char c : text) {
        const Glyph *glyph = FindGlyph(c);
        for (int row = 0; row < 7; ++row) for (int col = 0; col < 5; ++col) {
            if (!(glyph->rows[row] & (1U << (4 - col)))) continue;
            for (int sy = 0; sy < scale; ++sy) for (int sx = 0; sx < scale; ++sx) {
                const int px = x + col * scale + sx, py = y + row * scale + sy;
                if (px < width && py < height) PutPixel(frame, stride, px, py);
            }
        }
        x += 6 * scale;
    }
}

}

auto DebugOverlay::SetStreamInfo(std::string engineFileName, int inputWidth, int inputHeight,
                                 const char *inputFormat, bool bt709, bool fullRange,
                                 int outputWidth, int outputHeight) -> void {
    _engineLine = "ENGINE: " + std::move(engineFileName);
    char line[128] {};
    std::snprintf(line, sizeof(line), "INPUT: %dx%d %s, BT.%s %s",
                  inputWidth, inputHeight, inputFormat, bt709 ? "709" : "601",
                  fullRange ? "FULL" : "LIMITED");
    _inputLine = line;
    _transportLine = "TRANSPORT: HOST COPY";
    std::snprintf(line, sizeof(line), "OUTPUT: %dx%d", outputWidth, outputHeight);
    _outputLine = line;
}

auto DebugOverlay::SetTransportInfo(std::string transportLine) -> void {
    _transportLine = std::move(transportLine);
}

auto DebugOverlay::Draw(uint8_t *frame, size_t stride, int width, int height,
                        const DebugOverlayTimings &t) -> void {
    const std::array<double, 5> current {
        t.uploadMs, t.preprocessMs, t.inferenceMs, t.outputMs, t.pipelineMs
    };

    for (size_t i = 0; i < current.size(); ++i) {
        _averageSums[i] += current[i];
    }

    ++_averageCount;
    if (_averageCount == 1 && _displayedAverages[0] == 0) {
        // Provide useful values immediately while the first full batch is collected.
        _displayedAverages = current;
    }
    if (_averageCount == 24) {
        for (size_t i = 0; i < current.size(); ++i) {
            _displayedAverages[i] = _averageSums[i] / 24.0;
            _averageSums[i] = 0;
        }
        _averageCount = 0;
    }

    for (size_t i = 0; i < current.size(); ++i) {
        _maximums[i] = std::max(_maximums[i], current[i]);
    }

    ++_maximumUpdateCount;
    if (_maximumUpdateCount == 1 && _displayedMaximums[0] == 0) {
        _displayedMaximums = current;
    }
    if (_maximumUpdateCount == 120) {
        _displayedMaximums = _maximums;
        _maximums.fill(0);
        _maximumUpdateCount = 0;
    }

    constexpr int glyphAdvance = 6 * 3;
    const size_t longestLine = std::max<size_t>(_engineLine.size(), 76);
    const int desiredPanelWidth = 16 + static_cast<int>(longestLine) * glyphAdvance;
    const int panelWidth = std::min(width, desiredPanelWidth);
    const int panelHeight = std::min(height, 488);

    for (int y = 0; y < panelHeight; ++y) {
        std::memset(frame + static_cast<size_t>(y) * stride, 0,
                    static_cast<size_t>(panelWidth) * 3 * sizeof(uint16_t));
    }

    constexpr std::array<std::string_view, 5> labels {
        "UPLOAD", "PRE-PROCESS", "INFERENCE", "RGB48 OUTPUT", "PIPELINE"
    };

    char line[128] {};
    DrawText(frame, stride, width, height, 8, 8, "MLFILTER DEBUG");
    // One blank text row separates the heading from the stream information.
    DrawText(frame, stride, width, height, 8, 68, _engineLine);
    DrawText(frame, stride, width, height, 8, 98, _inputLine);
    DrawText(frame, stride, width, height, 8, 128, _outputLine);
    DrawText(frame, stride, width, height, 8, 158, _transportLine);

    std::snprintf(line, sizeof(line),
                  "MAPPED CACHE: %zu/%zu  TRANSIENT REGISTRATIONS %llu",
                  t.outputCacheSize, t.outputCacheCapacity,
                  static_cast<unsigned long long>(t.outputTransientTransfers));
    DrawText(frame, stride, width, height, 8, 188, line);

    // A blank row separates stream information from the live metrics.
    for (size_t i = 0; i < labels.size(); ++i) {
        std::snprintf(line, sizeof(line), "%-14s %7.3F MS (MAX %.3F)",
                      labels[i].data(), _displayedAverages[i], _displayedMaximums[i]);
        DrawText(frame, stride, width, height, 8, 248 + static_cast<int>(i) * 30, line);
    }
}

}
