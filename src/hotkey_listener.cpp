// SPDX-License-Identifier: Apache-2.0
#include "hotkey_listener.h"

#include <Windows.h>

namespace MLFilter {
namespace {
constexpr int OVERLAY_HOTKEY_ID = 1;
}

HotkeyListener::~HotkeyListener() { Stop(); }

auto HotkeyListener::Start(std::atomic<bool> &enabled) -> void {
    if (_thread.joinable()) {
        return;
    }

    {
        std::lock_guard lock(_readyMutex);
        _ready = false;
    }

    _thread = std::thread([this, &enabled]() {
        // Force creation of the thread's message queue before publishing its ID.
        MSG message {};
        PeekMessageW(&message, nullptr, WM_USER, WM_USER, PM_NOREMOVE);
        _threadId.store(GetCurrentThreadId(), std::memory_order_release);
        const bool registered = RegisterHotKey(nullptr, OVERLAY_HOTKEY_ID,
                                               MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, 'J') != FALSE;
        {
            std::lock_guard lock(_readyMutex);
            _ready = true;
        }
        _readyCondition.notify_one();
        if (registered) {
            while (GetMessageW(&message, nullptr, 0, 0) > 0) {
                if (message.message == WM_HOTKEY && message.wParam == OVERLAY_HOTKEY_ID) {
                    enabled.store(!enabled.load(std::memory_order_relaxed),
                                  std::memory_order_relaxed);
                }
            }
            UnregisterHotKey(nullptr, OVERLAY_HOTKEY_ID);
        }
        _threadId.store(0, std::memory_order_release);
    });

    std::unique_lock lock(_readyMutex);
    _readyCondition.wait(lock, [this]() { return _ready; });
}

auto HotkeyListener::Stop() -> void {
    if (!_thread.joinable()) {
        return;
    }

    const DWORD id = _threadId.load(std::memory_order_acquire);
    if (id != 0) {
        PostThreadMessageW(id, WM_QUIT, 0, 0);
    }
    _thread.join();
}

}
