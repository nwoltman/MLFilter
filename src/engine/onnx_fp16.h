// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace MLFilter {

struct OnnxFp16Result {
    bool ok = false;             // false only on a malformed/unsupported protobuf
    bool changed = false;        // true if any fp32 data was converted to fp16
    std::vector<uint8_t> bytes;  // converted model (or a copy of the input if unchanged)
    std::wstring message;
};

// Converts an fp32 ONNX model to fp16 by rewriting the serialized protobuf in place
// (no protobuf/onnx library dependency). Converts FLOAT initializers, constant/attribute
// tensors, graph input/output/value_info element types, and `Cast`-to-FLOAT attributes to
// their FLOAT16 equivalents; all other bytes are preserved verbatim.
//
// Intended for the common super-resolution CNN case. Models that store weights in external
// files are left unchanged (reported via `changed == false`).
auto ConvertOnnxToFp16(const std::vector<uint8_t> &input) -> OnnxFp16Result;

}
