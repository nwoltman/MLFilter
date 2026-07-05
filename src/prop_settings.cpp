// SPDX-License-Identifier: Apache-2.0

#include "prop_settings.h"

#include <array>
#include <filesystem>
#include <format>
#include <memory>
#include <thread>

#include <commdlg.h>

#include "constants.h"
#include "engine/engine_builder.h"
#include "engine/engine_naming.h"
#include "engine/tensorrt_engine_builder.h"
#include "resource.h"

#pragma comment(lib, "comdlg32")

namespace MLFilter {

namespace {

constexpr UINT WM_APP_LOG = WM_APP + 1;          // lParam: std::wstring* (heap-owned)
constexpr UINT WM_APP_BUILD_DONE = WM_APP + 2;

auto GetControlText(HWND dlg, int id) -> std::wstring {
    const int length = GetWindowTextLengthW(GetDlgItem(dlg, id));
    if (length <= 0) {
        return {};
    }
    std::wstring text(static_cast<size_t>(length), L'\0');
    GetDlgItemTextW(dlg, id, text.data(), length + 1);
    return text;
}

}

CMLFilterPropSettings::CMLFilterPropSettings(LPUNKNOWN pUnk, HRESULT *phr)
    : CBasePropertyPage(SETTINGS_PAGE_NAME, pUnk, IDD_SETTINGS_PAGE, IDS_SETTINGS) {}

auto CALLBACK CMLFilterPropSettings::CreateInstance(LPUNKNOWN pUnk, HRESULT *phr) -> CUnknown * {
    auto *instance = new CMLFilterPropSettings(pUnk, phr);
    if (instance == nullptr && phr != nullptr) {
        *phr = E_OUTOFMEMORY;
    }
    return instance;
}

auto CMLFilterPropSettings::OnActivate() -> HRESULT {
    _settings.Load();
    SetDlgItemTextW(m_Dlg, IDC_EDIT_MODEL, _settings.modelPath.c_str());
    SetDlgItemTextW(m_Dlg, IDC_EDIT_GLOBS, _settings.fileGlobs.c_str());
    CheckDlgButton(m_Dlg, IDC_CHECK_ONLY_1080P,
                   _settings.onlyRun1080pOrLower ? BST_CHECKED : BST_UNCHECKED);
    return S_OK;
}

auto CMLFilterPropSettings::OnApplyChanges() -> HRESULT {
    // Just persist settings. The 1080p engine is built automatically when a model is
    // selected (and can be triggered manually with the "Build 1080p engine" button).
    _settings.modelPath = GetControlText(m_Dlg, IDC_EDIT_MODEL);
    _settings.fileGlobs = GetControlText(m_Dlg, IDC_EDIT_GLOBS);
    _settings.onlyRun1080pOrLower =
        IsDlgButtonChecked(m_Dlg, IDC_CHECK_ONLY_1080P) == BST_CHECKED;
    _settings.Save();
    return S_OK;
}

auto CMLFilterPropSettings::SetDirty() -> void {
    m_bDirty = TRUE;
    if (m_pPageSite != nullptr) {
        m_pPageSite->OnStatusChange(PROPPAGESTATUS_DIRTY);
    }
}

auto CMLFilterPropSettings::AppendLog(const std::wstring &line) -> void {
    const HWND logControl = GetDlgItem(m_Dlg, IDC_EDIT_LOG);
    if (logControl == nullptr) {
        return;
    }

    const std::wstring withNewline = line + L"\r\n";
    const int end = GetWindowTextLengthW(logControl);
    SendMessageW(logControl, EM_SETSEL, end, end);
    SendMessageW(logControl, EM_REPLACESEL, FALSE, reinterpret_cast<LPARAM>(withNewline.c_str()));
}

auto CMLFilterPropSettings::StartBuild(bool quietIfNoModel) -> void {
    if (_isBuilding) {
        return;
    }

    const std::filesystem::path modelPath = GetControlText(m_Dlg, IDC_EDIT_MODEL);
    if (modelPath.empty() || !std::filesystem::exists(modelPath)) {
        if (!quietIfNoModel) {
            AppendLog(L"Please select a valid ONNX model file first.");
        }
        return;
    }

    const EngineBuildRequest request { .onnxPath = modelPath, .width = PREBUILD_WIDTH, .height = PREBUILD_HEIGHT };

    // Never rebuild an engine that already exists.
    TensorRTEngineBuilder builder;
    if (const std::filesystem::path existing = builder.EnginePath(request);
        !existing.empty() && std::filesystem::exists(existing)) {
        AppendLog(std::format(L"{}x{} engine is already built for this model: {}", PREBUILD_WIDTH, PREBUILD_HEIGHT, existing.filename().wstring()));
        AppendLog(L"Skipping build. Use \"Delete all engine files\" first if you want to rebuild it.");
        return;
    }

    _isBuilding = true;
    // Lock the model picker and build controls while the engine builds.
    EnableWindow(GetDlgItem(m_Dlg, IDC_BUTTON_BUILD), FALSE);
    EnableWindow(GetDlgItem(m_Dlg, IDC_EDIT_MODEL), FALSE);
    EnableWindow(GetDlgItem(m_Dlg, IDC_BUTTON_BROWSE), FALSE);
    EnableWindow(GetDlgItem(m_Dlg, IDC_BUTTON_DELETE_ENGINES), FALSE);

    const HWND dlg = m_Dlg;
    AppendLog(std::format(L"--- Building {}x{} engine ---", PREBUILD_WIDTH, PREBUILD_HEIGHT));

    // The worker captures only copies (no `this`), so it is safe even if the page is
    // closed mid-build. It reports progress and completion via PostMessage.
    std::thread([dlg, request]() {
        const ProgressFn progress = [dlg](const std::wstring &message) {
            PostMessageW(dlg, WM_APP_LOG, 0, reinterpret_cast<LPARAM>(new std::wstring(message)));
        };

        TensorRTEngineBuilder builder;
        const EngineBuildResult result = builder.Build(request, progress);
        if (!result.success) {
            progress(std::wstring(L"FAILED: ") + result.message);
        }

        PostMessageW(dlg, WM_APP_BUILD_DONE, 0, 0);
    }).detach();
}

auto CMLFilterPropSettings::DeleteAllEngines() -> void {
    if (_isBuilding) {
        AppendLog(L"Cannot delete engine files while a build is in progress.");
        return;
    }

    const std::filesystem::path dir = GetEnginesDirectory();
    if (dir.empty() || !std::filesystem::exists(dir)) {
        AppendLog(L"No engine cache directory found.");
        return;
    }

    if (MessageBoxW(m_Dlg, L"Delete all cached engine files? They will be rebuilt on demand.",
                    FILTER_NAME_WIDE, MB_YESNO | MB_ICONWARNING) != IDYES) {
        return;
    }

    int deleted = 0;
    std::error_code ec;
    for (const auto &entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) {
            break;
        }
        if (entry.is_regular_file() && entry.path().extension() == L".engine") {
            const std::wstring name = entry.path().filename().wstring();
            std::error_code removeEc;
            if (std::filesystem::remove(entry.path(), removeEc)) {
                AppendLog(std::format(L"Deleted engine: {}", name));
                ++deleted;
            }
        }
    }

    AppendLog(std::format(L"Deleted {} engine file(s).", deleted));
}

auto CMLFilterPropSettings::OnReceiveMessage(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) -> INT_PTR {
    switch (uMsg) {
    case WM_APP_LOG: {
        std::unique_ptr<std::wstring> line(reinterpret_cast<std::wstring *>(lParam));
        if (line) {
            AppendLog(*line);
        }
        return TRUE;
    }

    case WM_APP_BUILD_DONE:
        _isBuilding = false;
        EnableWindow(GetDlgItem(m_Dlg, IDC_BUTTON_BUILD), TRUE);
        EnableWindow(GetDlgItem(m_Dlg, IDC_EDIT_MODEL), TRUE);
        EnableWindow(GetDlgItem(m_Dlg, IDC_BUTTON_BROWSE), TRUE);
        EnableWindow(GetDlgItem(m_Dlg, IDC_BUTTON_DELETE_ENGINES), TRUE);
        AppendLog(L"--- Done ---");
        return TRUE;

    case WM_COMMAND:
        if (HIWORD(wParam) == BN_CLICKED) {
            switch (LOWORD(wParam)) {
            case IDC_BUTTON_BROWSE: {
                std::array<WCHAR, MAX_PATH> file {};

                OPENFILENAMEW ofn {};
                ofn.lStructSize = sizeof(ofn);
                ofn.hwndOwner = hwnd;
                ofn.lpstrFile = file.data();
                ofn.nMaxFile = static_cast<DWORD>(file.size());
                ofn.lpstrFilter = L"ONNX Models\0*.onnx\0All Files\0*.*\0";
                ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

                if (GetOpenFileNameW(&ofn) == TRUE) {
                    SetDlgItemTextW(hwnd, IDC_EDIT_MODEL, file.data());
                    SetDirty();
                    // Start building the 1080p engine immediately once a model is chosen.
                    StartBuild(/*quietIfNoModel*/ false);
                }
                return TRUE;
            }

            case IDC_BUTTON_BUILD:
                StartBuild(/*quietIfNoModel*/ false);
                return TRUE;

            case IDC_BUTTON_DELETE_ENGINES:
                DeleteAllEngines();
                return TRUE;

            case IDC_CHECK_ONLY_1080P:
                SetDirty();
                return TRUE;
            }
        } else if (HIWORD(wParam) == EN_CHANGE && (LOWORD(wParam) == IDC_EDIT_MODEL || LOWORD(wParam) == IDC_EDIT_GLOBS)) {
            SetDirty();
            return TRUE;
        }
        break;
    }

    return CBasePropertyPage::OnReceiveMessage(hwnd, uMsg, wParam, lParam);
}

}
