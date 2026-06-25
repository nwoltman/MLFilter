// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>

#include <windows.h>

namespace MLFilter {

// Thin RAII wrapper over an HKCU\Software\MLFilter registry key.
class Registry {
public:
    Registry() = default;
    Registry(const Registry &) = delete;
    Registry &operator=(const Registry &) = delete;
    ~Registry();

    auto Initialize() -> bool;
    explicit operator bool() const { return _registryKey != nullptr; }

    auto ReadString(const WCHAR *valueName) const -> std::wstring;
    auto ReadNumber(const WCHAR *valueName, int defaultValue) const -> DWORD;
    auto WriteString(const WCHAR *valueName, std::wstring_view valueString) const -> bool;
    auto WriteNumber(const WCHAR *valueName, DWORD valueNumber) const -> bool;

private:
    HKEY _registryKey = nullptr;
};

}
