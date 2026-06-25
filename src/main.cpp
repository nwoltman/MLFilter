// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include <streams.h>

#include "constants.h"
#include "filter.h"
#include "formats.h"
#include "guids.h"
#include "prop_settings.h"

#pragma comment(lib, "strmiids")
#pragma comment(lib, "winmm")

namespace MLFilter {

namespace {

// Media types are filled in at registration time from the shared SupportedSubtypes() list.
REGFILTERPINS REG_PINS[] {
    {
        .strName = nullptr,
        .bRendered = FALSE,
        .bOutput = FALSE,
        .bZero = FALSE,
        .bMany = FALSE,
        .clsConnectsToFilter = &CLSID_NULL,
        .strConnectsToPin = nullptr,
        .nMediaTypes = 0,
        .lpMediaType = nullptr,
    },
    {
        .strName = nullptr,
        .bRendered = FALSE,
        .bOutput = TRUE,
        .bZero = FALSE,
        .bMany = FALSE,
        .clsConnectsToFilter = &CLSID_NULL,
        .strConnectsToPin = nullptr,
        .nMediaTypes = 0,
        .lpMediaType = nullptr,
    },
};

AMOVIESETUP_FILTER REG_FILTER {
    .clsID = &CLSID_MLFilter,
    .strName = FILTER_NAME_WIDE,
    .dwMerit = MERIT_DO_NOT_USE + 1,
    .nPins = sizeof(REG_PINS) / sizeof(REG_PINS[0]),
    .lpPin = REG_PINS,
};

// Registers/unregisters the filter. On register, the pin media types are populated from
// SupportedSubtypes() so the graph builder (and MPC-BE's UI) see the exact decoded
// subtypes we accept, rather than "any video".
auto RegisterFilter(BOOL doRegister) -> HRESULT {
    if (!doRegister) {
        return AMovieDllRegisterServer2(FALSE);
    }

    const std::vector<GUID> &subtypes = SupportedSubtypes();
    std::vector<REGPINTYPES> pinTypes;
    pinTypes.reserve(subtypes.size());
    for (const GUID &subtype : subtypes) {
        pinTypes.push_back(REGPINTYPES { .clsMajorType = &MEDIATYPE_Video, .clsMinorType = &subtype });
    }

    for (REGFILTERPINS &pin : REG_PINS) {
        pin.nMediaTypes = static_cast<UINT>(pinTypes.size());
        pin.lpMediaType = pinTypes.data();
    }

    return AMovieDllRegisterServer2(TRUE);
}

}

}

CFactoryTemplate g_Templates[] {
    {
        .m_Name = MLFilter::FILTER_NAME_WIDE,
        .m_ClsID = &MLFilter::CLSID_MLFilter,
        .m_lpfnNew = MLFilter::CMLFilter::CreateInstance,
        .m_lpfnInit = nullptr,
        .m_pAMovieSetup_Filter = &MLFilter::REG_FILTER,
    },
    {
        .m_Name = MLFilter::SETTINGS_PAGE_NAME,
        .m_ClsID = &MLFilter::CLSID_MLFilterPropSettings,
        .m_lpfnNew = MLFilter::CMLFilterPropSettings::CreateInstance,
    },
};
int g_cTemplates = sizeof(g_Templates) / sizeof(g_Templates[0]);

auto STDAPICALLTYPE DllRegisterServer() -> HRESULT {
    return MLFilter::RegisterFilter(TRUE);
}

auto STDAPICALLTYPE DllUnregisterServer() -> HRESULT {
    return MLFilter::RegisterFilter(FALSE);
}

extern "C" DECLSPEC_NOINLINE auto WINAPI DllEntryPoint(HINSTANCE hInstance, ULONG ulReason, __inout_opt LPVOID pv) -> BOOL;

namespace {

// In a redistributable release the .ax sits next to a "bin" subfolder holding the
// bundled TensorRT + CUDA DLLs. Prepend it to the process PATH so the delay-loaded
// TensorRT DLLs — and the CUDA/builder-resource DLLs TensorRT loads dynamically — are
// found. No-op during development (no "bin" subfolder), where those DLLs are already on
// PATH via TENSORRT_ROOT/CUDA_PATH.
auto PrependBundledBinToPath(HMODULE module) -> void {
    std::array<wchar_t, MAX_PATH> modulePath {};
    const DWORD length = GetModuleFileNameW(module, modulePath.data(), static_cast<DWORD>(modulePath.size()));
    if (length == 0 || length >= modulePath.size()) {
        return;
    }

    const std::filesystem::path binDir = std::filesystem::path(modulePath.data()).parent_path() / L"bin";
    std::error_code ec;
    if (!std::filesystem::exists(binDir, ec)) {
        return;
    }

    std::wstring existingPath;
    if (const DWORD needed = GetEnvironmentVariableW(L"PATH", nullptr, 0); needed > 0) {
        existingPath.resize(needed);
        existingPath.resize(GetEnvironmentVariableW(L"PATH", existingPath.data(), needed));
    }

    const std::wstring newPath = binDir.wstring() + L";" + existingPath;
    SetEnvironmentVariableW(L"PATH", newPath.c_str());
}

}

auto APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) -> BOOL {
    if (ul_reason_for_call == DLL_PROCESS_ATTACH) {
        PrependBundledBinToPath(hModule);
    }
    return DllEntryPoint(static_cast<HINSTANCE>(hModule), ul_reason_for_call, lpReserved);
}
