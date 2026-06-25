// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <windows.h>

namespace MLFilter {

inline constexpr const char  *FILTER_NAME       = "MLFilter";
inline constexpr const WCHAR *FILTER_NAME_WIDE  = L"MLFilter";
inline constexpr const WCHAR *SETTINGS_PAGE_NAME = L"MLFilter Settings";

// HKCU\Software\MLFilter
inline constexpr const WCHAR *REGISTRY_KEY_NAME = L"Software\\MLFilter";

inline constexpr const WCHAR *SETTING_MODEL_PATH = L"ModelPath";
inline constexpr const WCHAR *SETTING_FILE_GLOBS = L"FileGlobs";

// Engines are built on demand from each video's actual resolution. This one
// resolution is always pre-built when a model is selected, so the most common case
// (1080p source) never waits on first play.
inline constexpr int PREBUILD_WIDTH = 1920;
inline constexpr int PREBUILD_HEIGHT = 1080;

}
