// SPDX-License-Identifier: Apache-2.0

#include "onnx_fp16.h"

#include <cstring>

// A surgical fp32 -> fp16 rewriter for ONNX models that operates directly on the
// serialized protobuf wire format, so it needs no protobuf or onnx library. Fields
// that are not part of the conversion are copied verbatim; only FLOAT tensor data,
// element types, and Cast-to-FLOAT targets are changed to their FLOAT16 equivalents.
//
// Relevant ONNX field numbers (from onnx.proto):
//   ModelProto.graph = 7
//   GraphProto: node=1, initializer=5, input=11, output=12, value_info=13
//   NodeProto: attribute=5
//   AttributeProto: i=3, t=5, tensors=10, name=1
//   ValueInfoProto.type=2; TypeProto.tensor_type=1; TypeProto.Tensor.elem_type=1
//   TensorProto: data_type=2, float_data=4, raw_data=9, external_data=13, data_location=14
//   TensorProto data types: FLOAT=1, FLOAT16=10

namespace MLFilter {

namespace {

constexpr uint64_t TP_FLOAT = 1;
constexpr uint64_t TP_FLOAT16 = 10;

constexpr uint32_t WT_VARINT = 0;
constexpr uint32_t WT_I64 = 1;
constexpr uint32_t WT_LEN = 2;
constexpr uint32_t WT_I32 = 5;

// IEEE-754 float -> half (round to nearest even), handling subnormals/inf/nan.
auto FloatToHalf(float value) -> uint16_t {
    uint32_t x;
    std::memcpy(&x, &value, sizeof(x));

    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exponent = (x >> 23) & 0xffu;
    uint32_t mantissa = x & 0x007fffffu;

    if (exponent == 0xff) {
        // Inf or NaN.
        return static_cast<uint16_t>(sign | 0x7c00u | (mantissa != 0 ? 0x0200u : 0u));
    }

    int32_t halfExp = static_cast<int32_t>(exponent) - 127 + 15;

    if (halfExp >= 0x1f) {
        return static_cast<uint16_t>(sign | 0x7c00u);  // overflow -> Inf
    }

    if (halfExp <= 0) {
        if (halfExp < -10) {
            return static_cast<uint16_t>(sign);  // underflow -> 0
        }
        mantissa |= 0x00800000u;
        const int shift = 14 - halfExp;
        uint32_t halfMant = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1);
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (halfMant & 1u))) {
            ++halfMant;
        }
        return static_cast<uint16_t>(sign | halfMant);
    }

    uint16_t half = static_cast<uint16_t>(sign | (static_cast<uint32_t>(halfExp) << 10) | (mantissa >> 13));
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) {
        ++half;  // round to nearest even (may carry into exponent, which is correct)
    }
    return half;
}

class Reader {
public:
    Reader(const uint8_t *data, size_t size)
        : _p(data)
        , _end(data + size) {}

    auto ok() const -> bool { return _ok; }
    auto eof() const -> bool { return _p >= _end; }

    auto ReadVarint(uint64_t &out) -> bool {
        uint64_t result = 0;
        int shift = 0;
        while (_p < _end) {
            const uint8_t byte = *_p++;
            result |= static_cast<uint64_t>(byte & 0x7f) << shift;
            if ((byte & 0x80) == 0) {
                out = result;
                return true;
            }
            shift += 7;
            if (shift > 63) {
                break;
            }
        }
        _ok = false;
        return false;
    }

    // Reads one field. Returns false at clean end-of-buffer or on malformed input
    // (distinguish with ok()).
    auto ReadField(uint32_t &field, uint32_t &wireType, uint64_t &varint, const uint8_t *&data, size_t &len) -> bool {
        if (eof()) {
            return false;
        }

        uint64_t tag = 0;
        if (!ReadVarint(tag)) {
            return false;
        }
        field = static_cast<uint32_t>(tag >> 3);
        wireType = static_cast<uint32_t>(tag & 7);
        varint = 0;
        data = nullptr;
        len = 0;

        switch (wireType) {
        case WT_VARINT:
            return ReadVarint(varint);
        case WT_I64:
            if (_end - _p < 8) {
                _ok = false;
                return false;
            }
            data = _p;
            len = 8;
            _p += 8;
            return true;
        case WT_I32:
            if (_end - _p < 4) {
                _ok = false;
                return false;
            }
            data = _p;
            len = 4;
            _p += 4;
            return true;
        case WT_LEN: {
            uint64_t payloadLen = 0;
            if (!ReadVarint(payloadLen)) {
                return false;
            }
            if (static_cast<uint64_t>(_end - _p) < payloadLen) {
                _ok = false;
                return false;
            }
            data = _p;
            len = static_cast<size_t>(payloadLen);
            _p += payloadLen;
            return true;
        }
        default:
            _ok = false;
            return false;
        }
    }

private:
    const uint8_t *_p;
    const uint8_t *_end;
    bool _ok = true;
};

auto PutVarint(std::vector<uint8_t> &out, uint64_t value) -> void {
    while (true) {
        const uint8_t byte = value & 0x7f;
        value >>= 7;
        if (value != 0) {
            out.push_back(byte | 0x80);
        } else {
            out.push_back(byte);
            break;
        }
    }
}

auto PutTag(std::vector<uint8_t> &out, uint32_t field, uint32_t wireType) -> void {
    PutVarint(out, (static_cast<uint64_t>(field) << 3) | wireType);
}

auto PutVarintField(std::vector<uint8_t> &out, uint32_t field, uint64_t value) -> void {
    PutTag(out, field, WT_VARINT);
    PutVarint(out, value);
}

auto PutLenField(std::vector<uint8_t> &out, uint32_t field, const uint8_t *data, size_t len) -> void {
    PutTag(out, field, WT_LEN);
    PutVarint(out, len);
    out.insert(out.end(), data, data + len);
}

// Re-emit a field exactly as it was read.
auto CopyField(std::vector<uint8_t> &out, uint32_t field, uint32_t wireType, uint64_t varint, const uint8_t *data, size_t len) -> void {
    switch (wireType) {
    case WT_VARINT:
        PutVarintField(out, field, varint);
        break;
    case WT_LEN:
        PutLenField(out, field, data, len);
        break;
    case WT_I64:
    case WT_I32:
        PutTag(out, field, wireType);
        out.insert(out.end(), data, data + len);
        break;
    default:
        break;
    }
}

struct RawField {
    uint32_t field;
    uint32_t wireType;
    uint64_t varint;
    const uint8_t *data;
    size_t len;
};

class Converter {
public:
    bool changed = false;
    bool ok = true;

    auto Model(const uint8_t *data, size_t len) -> std::vector<uint8_t> {
        std::vector<uint8_t> out;
        Reader reader(data, len);
        uint32_t field = 0, wireType = 0;
        uint64_t varint = 0;
        const uint8_t *payload = nullptr;
        size_t payloadLen = 0;
        while (reader.ReadField(field, wireType, varint, payload, payloadLen)) {
            if (field == 7 && wireType == WT_LEN) {  // graph
                const std::vector<uint8_t> graph = Graph(payload, payloadLen);
                PutLenField(out, field, graph.data(), graph.size());
            } else {
                CopyField(out, field, wireType, varint, payload, payloadLen);
            }
        }
        if (!reader.ok()) {
            ok = false;
        }
        return out;
    }

private:
    auto Graph(const uint8_t *data, size_t len) -> std::vector<uint8_t> {
        std::vector<uint8_t> out;
        Reader reader(data, len);
        uint32_t field = 0, wireType = 0;
        uint64_t varint = 0;
        const uint8_t *payload = nullptr;
        size_t payloadLen = 0;
        while (reader.ReadField(field, wireType, varint, payload, payloadLen)) {
            if (wireType == WT_LEN && field == 1) {  // node
                const std::vector<uint8_t> node = Node(payload, payloadLen);
                PutLenField(out, field, node.data(), node.size());
            } else if (wireType == WT_LEN && field == 5) {  // initializer
                const std::vector<uint8_t> tensor = Tensor(payload, payloadLen);
                PutLenField(out, field, tensor.data(), tensor.size());
            } else if (wireType == WT_LEN && (field == 11 || field == 12 || field == 13)) {  // input/output/value_info
                const std::vector<uint8_t> valueInfo = ValueInfo(payload, payloadLen);
                PutLenField(out, field, valueInfo.data(), valueInfo.size());
            } else {
                CopyField(out, field, wireType, varint, payload, payloadLen);
            }
        }
        if (!reader.ok()) {
            ok = false;
        }
        return out;
    }

    auto Node(const uint8_t *data, size_t len) -> std::vector<uint8_t> {
        std::vector<uint8_t> out;
        Reader reader(data, len);
        uint32_t field = 0, wireType = 0;
        uint64_t varint = 0;
        const uint8_t *payload = nullptr;
        size_t payloadLen = 0;
        while (reader.ReadField(field, wireType, varint, payload, payloadLen)) {
            if (wireType == WT_LEN && field == 5) {  // attribute
                const std::vector<uint8_t> attribute = Attribute(payload, payloadLen);
                PutLenField(out, field, attribute.data(), attribute.size());
            } else {
                CopyField(out, field, wireType, varint, payload, payloadLen);
            }
        }
        if (!reader.ok()) {
            ok = false;
        }
        return out;
    }

    auto Attribute(const uint8_t *data, size_t len) -> std::vector<uint8_t> {
        Reader scan(data, len);
        std::string name;
        uint32_t field = 0, wireType = 0;
        uint64_t varint = 0;
        const uint8_t *payload = nullptr;
        size_t payloadLen = 0;
        while (scan.ReadField(field, wireType, varint, payload, payloadLen)) {
            if (field == 1 && wireType == WT_LEN) {  // name
                name.assign(reinterpret_cast<const char *>(payload), payloadLen);
            }
        }
        if (!scan.ok()) {
            ok = false;
        }
        const bool isCastTo = name == "to";

        std::vector<uint8_t> out;
        Reader reader(data, len);
        while (reader.ReadField(field, wireType, varint, payload, payloadLen)) {
            if (wireType == WT_LEN && (field == 5 || field == 10)) {  // t / tensors
                const std::vector<uint8_t> tensor = Tensor(payload, payloadLen);
                PutLenField(out, field, tensor.data(), tensor.size());
            } else if (isCastTo && field == 3 && wireType == WT_VARINT && varint == TP_FLOAT) {  // Cast "to" = FLOAT
                PutVarintField(out, field, TP_FLOAT16);
                changed = true;
            } else {
                CopyField(out, field, wireType, varint, payload, payloadLen);
            }
        }
        return out;
    }

    auto ValueInfo(const uint8_t *data, size_t len) -> std::vector<uint8_t> {
        std::vector<uint8_t> out;
        Reader reader(data, len);
        uint32_t field = 0, wireType = 0;
        uint64_t varint = 0;
        const uint8_t *payload = nullptr;
        size_t payloadLen = 0;
        while (reader.ReadField(field, wireType, varint, payload, payloadLen)) {
            if (field == 2 && wireType == WT_LEN) {  // type
                const std::vector<uint8_t> type = TypeProto(payload, payloadLen);
                PutLenField(out, field, type.data(), type.size());
            } else {
                CopyField(out, field, wireType, varint, payload, payloadLen);
            }
        }
        if (!reader.ok()) {
            ok = false;
        }
        return out;
    }

    auto TypeProto(const uint8_t *data, size_t len) -> std::vector<uint8_t> {
        std::vector<uint8_t> out;
        Reader reader(data, len);
        uint32_t field = 0, wireType = 0;
        uint64_t varint = 0;
        const uint8_t *payload = nullptr;
        size_t payloadLen = 0;
        while (reader.ReadField(field, wireType, varint, payload, payloadLen)) {
            if (field == 1 && wireType == WT_LEN) {  // tensor_type
                const std::vector<uint8_t> tensorType = TensorType(payload, payloadLen);
                PutLenField(out, field, tensorType.data(), tensorType.size());
            } else {
                CopyField(out, field, wireType, varint, payload, payloadLen);
            }
        }
        if (!reader.ok()) {
            ok = false;
        }
        return out;
    }

    auto TensorType(const uint8_t *data, size_t len) -> std::vector<uint8_t> {
        std::vector<uint8_t> out;
        Reader reader(data, len);
        uint32_t field = 0, wireType = 0;
        uint64_t varint = 0;
        const uint8_t *payload = nullptr;
        size_t payloadLen = 0;
        while (reader.ReadField(field, wireType, varint, payload, payloadLen)) {
            if (field == 1 && wireType == WT_VARINT && varint == TP_FLOAT) {  // elem_type
                PutVarintField(out, field, TP_FLOAT16);
                changed = true;
            } else {
                CopyField(out, field, wireType, varint, payload, payloadLen);
            }
        }
        if (!reader.ok()) {
            ok = false;
        }
        return out;
    }

    auto Tensor(const uint8_t *data, size_t len) -> std::vector<uint8_t> {
        Reader reader(data, len);
        std::vector<RawField> fields;
        uint64_t dataType = 0;
        bool hasExternalData = false;

        uint32_t field = 0, wireType = 0;
        uint64_t varint = 0;
        const uint8_t *payload = nullptr;
        size_t payloadLen = 0;
        while (reader.ReadField(field, wireType, varint, payload, payloadLen)) {
            fields.push_back({ field, wireType, varint, payload, payloadLen });
            if (field == 2 && wireType == WT_VARINT) {
                dataType = varint;
            } else if (field == 13) {  // external_data
                hasExternalData = true;
            } else if (field == 14 && wireType == WT_VARINT && varint != 0) {  // data_location != DEFAULT
                hasExternalData = true;
            }
        }
        if (!reader.ok()) {
            ok = false;
        }

        std::vector<uint8_t> out;

        if (dataType != TP_FLOAT || hasExternalData) {
            for (const RawField &f : fields) {
                CopyField(out, f.field, f.wireType, f.varint, f.data, f.len);
            }
            return out;
        }

        // Gather the fp32 values from raw_data (9) and/or float_data (4).
        std::vector<float> values;
        for (const RawField &f : fields) {
            if (f.field == 9 && f.wireType == WT_LEN) {  // raw_data
                const size_t count = f.len / 4;
                for (size_t i = 0; i < count; ++i) {
                    float value = 0.0f;
                    std::memcpy(&value, f.data + i * 4, 4);
                    values.push_back(value);
                }
            } else if (f.field == 4 && f.wireType == WT_LEN) {  // packed float_data
                const size_t count = f.len / 4;
                for (size_t i = 0; i < count; ++i) {
                    float value = 0.0f;
                    std::memcpy(&value, f.data + i * 4, 4);
                    values.push_back(value);
                }
            } else if (f.field == 4 && f.wireType == WT_I32) {  // unpacked float_data element
                float value = 0.0f;
                std::memcpy(&value, f.data, 4);
                values.push_back(value);
            }
        }

        std::vector<uint8_t> halfData;
        halfData.reserve(values.size() * 2);
        for (const float value : values) {
            const uint16_t half = FloatToHalf(value);
            halfData.push_back(static_cast<uint8_t>(half & 0xff));
            halfData.push_back(static_cast<uint8_t>(half >> 8));
        }

        // Re-emit: data_type becomes FLOAT16, fp32 data fields are dropped, and the
        // converted values are written back as raw_data.
        for (const RawField &f : fields) {
            if (f.field == 2 && f.wireType == WT_VARINT) {
                PutVarintField(out, 2, TP_FLOAT16);
            } else if (f.field == 4 || f.field == 9) {
                // dropped; raw_data re-added below
            } else {
                CopyField(out, f.field, f.wireType, f.varint, f.data, f.len);
            }
        }
        if (!halfData.empty()) {
            PutLenField(out, 9, halfData.data(), halfData.size());
        }
        changed = true;
        return out;
    }
};

}

auto ConvertOnnxToFp16(const std::vector<uint8_t> &input) -> OnnxFp16Result {
    OnnxFp16Result result;

    if (input.empty()) {
        result.message = L"Empty ONNX model buffer.";
        return result;
    }

    Converter converter;
    std::vector<uint8_t> out = converter.Model(input.data(), input.size());

    if (!converter.ok) {
        // Malformed/unsupported: hand back the original untouched so the caller can
        // still attempt to parse it.
        result.ok = false;
        result.changed = false;
        result.bytes = input;
        result.message = L"Could not parse the ONNX protobuf for fp16 conversion; using the model as-is.";
        return result;
    }

    result.ok = true;
    result.changed = converter.changed;
    result.bytes = std::move(out);
    return result;
}

}
