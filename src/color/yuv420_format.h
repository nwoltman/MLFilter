// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace MLFilter {

// Backend-neutral description of the decoded input frame. CUDA uses it today; a future DirectML
// inference session can consume the same contract without changing FrameProcessor.
enum class Yuv420Format {
    NV12,
    P010,
};

struct Yuv420Conversion {
    Yuv420Format format;
    bool bt709;
    bool fullRange;
};

}
