// SPDX-License-Identifier: Apache-2.0

#include "progress_window.h"

#include <memory>

// g_hInst is defined by the DirectShow base classes (dllentry.cpp) and is the
// module instance handle used for window-class registration.
extern HINSTANCE g_hInst;

namespace MLFilter {

namespace {

constexpr UINT WM_APP_LOG = WM_APP + 1;     // lParam: std::wstring* (heap-owned)
constexpr UINT WM_APP_CLOSE = WM_APP + 2;
constexpr int IDC_LOG_EDIT = 1001;
constexpr wchar_t WINDOW_CLASS_NAME[] = L"MLFilterProgressWindow";

auto EnsureWindowClass() -> void {
    static bool registered = false;
    if (registered) {
        return;
    }
    registered = true;

    WNDCLASSEXW wc {};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = &ProgressWindow::WndProc;
    wc.hInstance = g_hInst;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_BTNFACE + 1);
    wc.lpszClassName = WINDOW_CLASS_NAME;
    RegisterClassExW(&wc);
}

}

ProgressWindow::~ProgressWindow() {
    Close();
}

auto ProgressWindow::Open(const std::wstring &title) -> void {
    if (_thread.joinable()) {
        return;
    }
    _readyEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    _thread = std::thread(&ProgressWindow::ThreadMain, this, title);
    if (_readyEvent != nullptr) {
        WaitForSingleObject(_readyEvent, 5000);
    }
}

auto ProgressWindow::Log(const std::wstring &line) -> void {
    if (_hwnd != nullptr) {
        PostMessageW(_hwnd, WM_APP_LOG, 0, reinterpret_cast<LPARAM>(new std::wstring(line)));
    }
}

auto ProgressWindow::Close() -> void {
    if (_hwnd != nullptr) {
        PostMessageW(_hwnd, WM_APP_CLOSE, 0, 0);
    }
    if (_thread.joinable()) {
        _thread.join();
    }
    if (_readyEvent != nullptr) {
        CloseHandle(_readyEvent);
        _readyEvent = nullptr;
    }
    _hwnd = nullptr;
}

auto ProgressWindow::ThreadMain(std::wstring title) -> void {
    EnsureWindowClass();

    constexpr int width = 540;
    constexpr int height = 260;
    const int x = (GetSystemMetrics(SM_CXSCREEN) - width) / 2;
    const int y = (GetSystemMetrics(SM_CYSCREEN) - height) / 2;

    _hwnd = CreateWindowExW(WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
                            WINDOW_CLASS_NAME,
                            title.c_str(),
                            WS_OVERLAPPED | WS_CAPTION | WS_BORDER,
                            x, y, width, height,
                            nullptr, nullptr, g_hInst, nullptr);

    if (_readyEvent != nullptr) {
        SetEvent(_readyEvent);
    }

    if (_hwnd == nullptr) {
        return;
    }

    ShowWindow(_hwnd, SW_SHOWNORMAL);
    UpdateWindow(_hwnd);

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}

auto CALLBACK ProgressWindow::WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) -> LRESULT {
    switch (msg) {
    case WM_CREATE: {
        RECT rc;
        GetClientRect(hwnd, &rc);
        const HWND edit = CreateWindowExW(0, L"EDIT", nullptr,
                                          WS_CHILD | WS_VISIBLE | ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | WS_VSCROLL,
                                          0, 0, rc.right, rc.bottom,
                                          hwnd, reinterpret_cast<HMENU>(static_cast<UINT_PTR>(IDC_LOG_EDIT)), g_hInst, nullptr);
        SendMessageW(edit, WM_SETFONT, reinterpret_cast<WPARAM>(GetStockObject(DEFAULT_GUI_FONT)), TRUE);
        return 0;
    }

    case WM_SIZE: {
        const HWND edit = GetDlgItem(hwnd, IDC_LOG_EDIT);
        if (edit != nullptr) {
            MoveWindow(edit, 0, 0, LOWORD(lParam), HIWORD(lParam), TRUE);
        }
        return 0;
    }

    case WM_APP_LOG: {
        std::unique_ptr<std::wstring> line(reinterpret_cast<std::wstring *>(lParam));
        const HWND edit = GetDlgItem(hwnd, IDC_LOG_EDIT);
        if (line && edit != nullptr) {
            const std::wstring withNewline = *line + L"\r\n";
            const int end = GetWindowTextLengthW(edit);
            SendMessageW(edit, EM_SETSEL, end, end);
            SendMessageW(edit, EM_REPLACESEL, FALSE, reinterpret_cast<LPARAM>(withNewline.c_str()));
        }
        return 0;
    }

    case WM_APP_CLOSE:
        DestroyWindow(hwnd);
        return 0;

    case WM_CLOSE:
        // Ignore the user's close request; the owning operation controls the lifetime.
        return 0;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }

    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

}
