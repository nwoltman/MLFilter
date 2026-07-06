// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <windows.h>

namespace MLFilter {

inline constexpr const char  *FILTER_NAME       = "MLFilter";
inline constexpr const WCHAR *FILTER_NAME_WIDE  = L"MLFilter";
inline constexpr const WCHAR *SETTINGS_PAGE_NAME = L"MLFilter Settings";

// HKCU\Software\MLFilter
inline constexpr const WCHAR *REGISTRY_KEY_NAME = L"Software\\MLFilter";

inline constexpr const WCHAR *SETTING_HD_MODEL_PATH = L"HDModelPath";
inline constexpr const WCHAR *SETTING_SD_MODEL_PATH = L"SDModelPath";
inline constexpr const WCHAR *SETTING_FILE_GLOBS = L"FileGlobs";
inline constexpr const WCHAR *SETTING_ONLY_RUN_1080P_OR_LOWER = L"OnlyRun1080pOrLower";

// Engines are built on demand from each video's actual resolution. This one
// resolution is always pre-built when a model is selected, so the most common case
// (1080p source) never waits on first play.
inline constexpr int HD_PREBUILD_WIDTH = 1920;
inline constexpr int HD_PREBUILD_HEIGHT = 1080;
inline constexpr int SD_PREBUILD_WIDTH = 720;
inline constexpr int SD_PREBUILD_HEIGHT = 480;
inline constexpr int HD_MIN_WIDTH = 1280;
inline constexpr int HD_MIN_HEIGHT = 720;

// Inputs above 1080p are left untouched and MLFilter removes itself from the graph.
inline constexpr int MAX_INPUT_WIDTH = 1920;
inline constexpr int MAX_INPUT_HEIGHT = 1080;

}
