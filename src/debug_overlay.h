// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace MLFilter {

struct DebugOverlayTimings {
    double uploadMs, preprocessMs, inferenceMs, packMs, downloadMs, outputRegistrationMs;
    size_t outputCacheSize, outputCacheCapacity;
    uint64_t outputTransientTransfers, outputRegistrationFailures;
};

class DebugOverlay {
public:
    auto SetStreamInfo(std::string engineFileName, int inputWidth, int inputHeight,
                       const char *inputFormat, bool bt709, bool fullRange,
                       int outputWidth, int outputHeight) -> void;
    auto Draw(uint8_t *frame, size_t strideBytes, int width, int height,
              const DebugOverlayTimings &timings) -> void;

private:
    std::array<double, 7> _averageSums {};
    std::array<double, 7> _displayedAverages {};
    size_t _averageCount = 0;
    std::array<double, 7> _maximums {};
    std::array<double, 7> _displayedMaximums {};
    size_t _maximumUpdateCount = 0;
    std::string _engineLine;
    std::string _inputLine;
    std::string _outputLine;
};

}
