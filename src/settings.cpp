// SPDX-License-Identifier: Apache-2.0

#include "settings.h"

#include "constants.h"
#include "registry.h"

namespace MLFilter {

auto Settings::Load() -> void {
    Registry registry;
    if (!registry.Initialize()) {
        return;
    }

    hdModelPath = registry.ReadString(SETTING_HD_MODEL_PATH);
    sdModelPath = registry.ReadString(SETTING_SD_MODEL_PATH);
    fileGlobs = registry.ReadString(SETTING_FILE_GLOBS);
    onlyRun1080pOrLower = registry.ReadNumber(SETTING_ONLY_RUN_1080P_OR_LOWER, 1) != 0;
}

auto Settings::Save() const -> void {
    Registry registry;
    if (!registry.Initialize()) {
        return;
    }

    registry.WriteString(SETTING_HD_MODEL_PATH, hdModelPath);
    registry.WriteString(SETTING_SD_MODEL_PATH, sdModelPath);
    registry.WriteString(SETTING_FILE_GLOBS, fileGlobs);
    registry.WriteNumber(SETTING_ONLY_RUN_1080P_OR_LOWER, onlyRun1080pOrLower ? 1 : 0);
}

}
