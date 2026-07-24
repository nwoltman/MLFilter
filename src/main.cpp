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

// Media types are filled in at registration time.
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

// Registers/unregisters the filter. The wildcard input subtype prevents an upstream decoder
// from converting its native output merely to satisfy MLFilter's registration metadata.
// MLFilter reports an error while streaming if the connected subtype is not internally supported.
auto RegisterFilter(BOOL doRegister) -> HRESULT {
    if (!doRegister) {
        return AMovieDllRegisterServer2(FALSE);
    }

    static const REGPINTYPES inputType { .clsMajorType = &MEDIATYPE_Video, .clsMinorType = &GUID_NULL };
    static const REGPINTYPES outputType { .clsMajorType = &MEDIATYPE_Video, .clsMinorType = &OutputSubtype() };

    REG_PINS[0].nMediaTypes = 1;
    REG_PINS[0].lpMediaType = &inputType;
    REG_PINS[1].nMediaTypes = 1;
    REG_PINS[1].lpMediaType = &outputType;

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

// In a redistributable release the .ax and its bundled TensorRT + CUDA DLLs all sit
// in a "bin" folder. Prepend it to the process PATH so the delay-loaded TensorRT DLLs
// and the CUDA/builder-resource DLLs TensorRT loads dynamically are found. No-op
// during development, where the .ax is not in a "bin" folder and those DLLs are
// already on PATH via TENSORRT_ROOT/CUDA_PATH.
auto PrependBundledBinToPath(HMODULE module) -> void {
    std::array<wchar_t, MAX_PATH> modulePath {};
    const DWORD length = GetModuleFileNameW(module, modulePath.data(), static_cast<DWORD>(modulePath.size()));
    if (length == 0 || length >= modulePath.size()) {
        return;
    }

    const std::filesystem::path binDir = std::filesystem::path(modulePath.data()).parent_path();
    if (binDir.filename() != L"bin") {
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
