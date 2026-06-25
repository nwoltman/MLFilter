// SPDX-License-Identifier: Apache-2.0

#include "registry.h"

#include "constants.h"

namespace MLFilter {

Registry::~Registry() {
    if (_registryKey) {
        RegCloseKey(_registryKey);
    }
}

auto Registry::Initialize() -> bool {
    return RegCreateKeyExW(HKEY_CURRENT_USER, REGISTRY_KEY_NAME, 0, nullptr, 0, KEY_QUERY_VALUE | KEY_SET_VALUE, nullptr, &_registryKey, nullptr) == ERROR_SUCCESS;
}

auto Registry::ReadString(const WCHAR *valueName) const -> std::wstring {
    if (_registryKey == nullptr) {
        return {};
    }

    // Query the required size first so arbitrarily long values (e.g. a glob list or a
    // path longer than MAX_PATH) are read in full.
    DWORD byteSize = 0;
    if (RegGetValueW(_registryKey, nullptr, valueName, RRF_RT_REG_SZ, nullptr, nullptr, &byteSize) != ERROR_SUCCESS || byteSize < sizeof(WCHAR)) {
        return {};
    }

    std::wstring buffer(byteSize / sizeof(WCHAR), L'\0');
    if (RegGetValueW(_registryKey, nullptr, valueName, RRF_RT_REG_SZ, nullptr, buffer.data(), &byteSize) != ERROR_SUCCESS) {
        return {};
    }

    buffer.resize(byteSize / sizeof(WCHAR) - 1);  // drop the trailing null
    return buffer;
}

auto Registry::ReadNumber(const WCHAR *valueName, int defaultValue) const -> DWORD {
    DWORD ret = static_cast<DWORD>(defaultValue);

    if (_registryKey) {
        DWORD valueSize = sizeof(ret);
        RegGetValueW(_registryKey, nullptr, valueName, RRF_RT_REG_DWORD, nullptr, &ret, &valueSize);
    }

    return ret;
}

auto Registry::WriteString(const WCHAR *valueName, std::wstring_view valueString) const -> bool {
    return _registryKey && RegSetValueExW(_registryKey,
                                          valueName,
                                          0,
                                          REG_SZ,
                                          reinterpret_cast<const BYTE *>(valueString.data()),
                                          static_cast<DWORD>((valueString.size() + 1) * sizeof(WCHAR))) == ERROR_SUCCESS;
}

auto Registry::WriteNumber(const WCHAR *valueName, DWORD valueNumber) const -> bool {
    return _registryKey && RegSetValueExW(_registryKey, valueName, 0, REG_DWORD, reinterpret_cast<const BYTE *>(&valueNumber), sizeof(valueNumber)) == ERROR_SUCCESS;
}

}
