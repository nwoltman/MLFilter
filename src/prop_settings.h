// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include <streams.h>

#include "settings.h"

namespace MLFilter {

// Property page: select an ONNX model, choose input resolutions, and build a
// static fp16 TensorRT engine per selected resolution.
class CMLFilterPropSettings : public CBasePropertyPage {
public:
    CMLFilterPropSettings(LPUNKNOWN pUnk, HRESULT *phr);

    static auto CALLBACK CreateInstance(LPUNKNOWN pUnk, HRESULT *phr) -> CUnknown *;

    auto OnActivate() -> HRESULT override;
    auto OnApplyChanges() -> HRESULT override;
    auto OnReceiveMessage(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) -> INT_PTR override;

private:
    auto SetDirty() -> void;
    auto StartBuild(bool quietIfNoModel) -> void;
    auto DeleteAllEngines() -> void;
    auto AppendLog(const std::wstring &line) -> void;

    Settings _settings;
    bool _isBuilding = false;
};

}
