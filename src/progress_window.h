// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <string>
#include <thread>

#include <windows.h>

namespace MLFilter {

// A small, top-most status window that runs on its own UI thread, so it stays
// responsive (repaints, can be moved) even while the thread that created it is
// blocked doing synchronous work such as building an engine.
class ProgressWindow {
public:
    ProgressWindow() = default;
    ProgressWindow(const ProgressWindow &) = delete;
    ProgressWindow &operator=(const ProgressWindow &) = delete;
    ~ProgressWindow();

    // Creates and shows the window; blocks until it exists. Safe to call once.
    auto Open(const std::wstring &title) -> void;

    // Appends a line to the window's log (thread-safe; posts to the window thread).
    auto Log(const std::wstring &line) -> void;

    // Closes the window and joins its thread. Safe to call multiple times.
    auto Close() -> void;

private:
    auto ThreadMain(std::wstring title) -> void;
    static auto CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) -> LRESULT;

    std::thread _thread;
    HWND _hwnd = nullptr;
    HANDLE _readyEvent = nullptr;
};

}
