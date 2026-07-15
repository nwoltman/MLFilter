// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace MLFilter {

struct DebugOverlayTimings {
    double pipelineMs;
    size_t outputCacheSize, outputCacheCapacity;
    uint64_t outputTransientTransfers;
};

class DebugOverlay {
public:
    auto SetStreamInfo(std::string engineFileName, int inputWidth, int inputHeight,
                       const char *inputFormat, bool bt709, bool fullRange,
                       int outputWidth, int outputHeight) -> void;
    auto SetTransportInfo(std::string transportLine) -> void;
    auto Draw(uint8_t *frame, size_t strideBytes, int width, int height,
              const DebugOverlayTimings &timings) -> void;

private:
    double _averageSum = 0;
    double _displayedAverage = 0;
    size_t _averageCount = 0;
    double _maximum = 0;
    double _displayedMaximum = 0;
    size_t _maximumUpdateCount = 0;
    std::string _engineLine;
    std::string _inputLine;
    std::string _outputLine;
    std::string _transportLine;
};

}
