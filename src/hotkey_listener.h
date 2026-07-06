// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace MLFilter {

// Registers Ctrl+Alt+J on a sleeping message thread, keeping keyboard polling out of Transform().
class HotkeyListener {
public:
    ~HotkeyListener();
    auto Start(std::atomic<bool> &enabled) -> void;
    auto Stop() -> void;

    HotkeyListener() = default;
    HotkeyListener(const HotkeyListener &) = delete;
    auto operator=(const HotkeyListener &) -> HotkeyListener & = delete;

private:
    std::thread _thread;
    std::atomic<unsigned long> _threadId = 0;
    std::mutex _readyMutex;
    std::condition_variable _readyCondition;
    bool _ready = false;
};

}
