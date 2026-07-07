// SPDX-License-Identifier: Apache-2.0

#include "d3d11_cuda_input.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cuda_d3d11_interop.h>
#include <cuda_runtime.h>
#include <d3d11.h>
#include <wrl/client.h>

#include "cuda_raii.h"
#include "fp16_kernels.h"
#include "shaders/pack_planes_nv12_bytecode.h"
#include "shaders/pack_planes_p010_bytecode.h"

namespace MLFilter {
namespace {

using Microsoft::WRL::ComPtr;

class D3D11ContextGuard {
public:
    explicit D3D11ContextGuard(HANDLE mutex) : _mutex(mutex) {
        if (_mutex != nullptr) {
            const DWORD wait = WaitForSingleObject(_mutex, INFINITE);
            _locked = wait == WAIT_OBJECT_0 || wait == WAIT_ABANDONED;
        }
    }

    ~D3D11ContextGuard() {
        if (_locked) {
            ReleaseMutex(_mutex);
        }
    }

    auto Locked() const -> bool { return _mutex == nullptr || _locked; }

private:
    HANDLE _mutex = nullptr;
    bool _locked = false;
};

class D3D11ComputeStateGuard {
public:
    explicit D3D11ComputeStateGuard(ID3D11DeviceContext *context)
        : _context(context) {
        std::array<ID3D11ClassInstance *, D3D11_SHADER_MAX_INTERFACES> classInstances {};
        _classInstanceCount = static_cast<UINT>(_classInstances.size());
        _context->CSGetShader(_shader.ReleaseAndGetAddressOf(), classInstances.data(),
                              &_classInstanceCount);
        for (UINT index = 0; index < _classInstanceCount; ++index) {
            _classInstances[index].Attach(classInstances[index]);
        }

        _context->CSGetConstantBuffers(0, 1, _constantBuffer.ReleaseAndGetAddressOf());

        std::array<ID3D11ShaderResourceView *, 2> shaderResources {};
        _context->CSGetShaderResources(0, static_cast<UINT>(_shaderResources.size()),
                                       shaderResources.data());
        for (size_t index = 0; index < _shaderResources.size(); ++index) {
            _shaderResources[index].Attach(shaderResources[index]);
        }

        _context->CSGetUnorderedAccessViews(0, 1,
                                            _unorderedAccessView.ReleaseAndGetAddressOf());
    }

    ~D3D11ComputeStateGuard() {
        std::array<ID3D11ShaderResourceView *, 2> nullShaderResources {};
        ID3D11UnorderedAccessView *nullUnorderedAccessView = nullptr;
        _context->CSSetShaderResources(0, static_cast<UINT>(nullShaderResources.size()),
                                       nullShaderResources.data());
        _context->CSSetUnorderedAccessViews(0, 1, &nullUnorderedAccessView, nullptr);

        std::array<ID3D11ClassInstance *, D3D11_SHADER_MAX_INTERFACES> classInstances {};
        for (UINT index = 0; index < _classInstanceCount; ++index) {
            classInstances[index] = _classInstances[index].Get();
        }

        ID3D11ClassInstance **classInstanceData = _classInstanceCount != 0
            ? classInstances.data()
            : nullptr;
        _context->CSSetShader(_shader.Get(), classInstanceData, _classInstanceCount);

        ID3D11Buffer *constantBuffer = _constantBuffer.Get();
        _context->CSSetConstantBuffers(0, 1, &constantBuffer);

        std::array<ID3D11ShaderResourceView *, 2> shaderResources {};
        for (size_t index = 0; index < _shaderResources.size(); ++index) {
            shaderResources[index] = _shaderResources[index].Get();
        }
        _context->CSSetShaderResources(0, static_cast<UINT>(_shaderResources.size()),
                                       shaderResources.data());

        ID3D11UnorderedAccessView *unorderedAccessView = _unorderedAccessView.Get();
        _context->CSSetUnorderedAccessViews(0, 1, &unorderedAccessView, nullptr);
    }

    D3D11ComputeStateGuard(const D3D11ComputeStateGuard &) = delete;
    auto operator=(const D3D11ComputeStateGuard &) -> D3D11ComputeStateGuard & = delete;

private:
    ID3D11DeviceContext *_context;
    ComPtr<ID3D11ComputeShader> _shader;
    std::array<ComPtr<ID3D11ClassInstance>, D3D11_SHADER_MAX_INTERFACES> _classInstances;
    UINT _classInstanceCount = 0;
    ComPtr<ID3D11Buffer> _constantBuffer;
    std::array<ComPtr<ID3D11ShaderResourceView>, 2> _shaderResources;
    ComPtr<ID3D11UnorderedAccessView> _unorderedAccessView;
};

struct D3D11Yuv420FormatInfo {
    DXGI_FORMAT sourceFormat;
    DXGI_FORMAT sourceYViewFormat;
    DXGI_FORMAT sourceUvViewFormat;
    UINT sampleBytes;
};

auto GetD3D11Yuv420FormatInfo(Yuv420Format format) -> D3D11Yuv420FormatInfo {
    switch (format) {
        case Yuv420Format::NV12:
            return {
                DXGI_FORMAT_NV12,
                DXGI_FORMAT_R8_UNORM,
                DXGI_FORMAT_R8G8_UNORM,
                1
            };
        case Yuv420Format::P010:
            return {
                DXGI_FORMAT_P010,
                DXGI_FORMAT_R16_UNORM,
                DXGI_FORMAT_R16G16_UNORM,
                2
            };
        default:
            throw std::invalid_argument("[GetD3D11Yuv420FormatInfo] Received invalid format value");
    }
}

auto CreatePackedBuffer(ID3D11Device *device, UINT byteWidth, ID3D11Buffer **buffer) -> HRESULT {
    D3D11_BUFFER_DESC desc {};
    desc.ByteWidth = byteWidth;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    return device->CreateBuffer(&desc, nullptr, buffer);
}

auto CreatePackedBufferUav(ID3D11Device *device, ID3D11Buffer *buffer, UINT byteWidth,
                           ID3D11UnorderedAccessView **uav) -> HRESULT {
    D3D11_UNORDERED_ACCESS_VIEW_DESC desc {};
    desc.Format = DXGI_FORMAT_R32_TYPELESS;
    desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    desc.Buffer.NumElements = byteWidth / sizeof(UINT);
    desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
    return device->CreateUnorderedAccessView(buffer, &desc, uav);
}

auto CreateSourceSrv(ID3D11Device *device, ID3D11Texture2D *texture, DXGI_FORMAT format,
                     unsigned arraySlice, ID3D11ShaderResourceView **srv) -> HRESULT {
    D3D11_SHADER_RESOURCE_VIEW_DESC desc {};
    desc.Format = format;
    desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
    desc.Texture2DArray.MipLevels = 1;
    desc.Texture2DArray.ArraySize = 1;
    desc.Texture2DArray.FirstArraySlice = arraySlice;
    return device->CreateShaderResourceView(texture, &desc, srv);
}

auto CreatePackPlanesShader(ID3D11Device *device, Yuv420Format format,
                            ID3D11ComputeShader **shader) -> HRESULT {
    switch (format) {
        case Yuv420Format::NV12:
            return device->CreateComputeShader(
                kPackPlanesNv12Bytecode, sizeof(kPackPlanesNv12Bytecode), nullptr, shader);
        case Yuv420Format::P010:
            return device->CreateComputeShader(
                kPackPlanesP010Bytecode, sizeof(kPackPlanesP010Bytecode), nullptr, shader);
        default:
            return E_INVALIDARG;
    }
}

struct PlaneCopyConstants {
    UINT width;
    UINT height;
    UINT rowPitch;
    UINT yPlaneBytes;
};

static_assert(sizeof(PlaneCopyConstants) == 16);

}

struct D3D11CudaInput::Impl {
    struct SourcePlaneSrvs {
        ComPtr<ID3D11ShaderResourceView> y;
        ComPtr<ID3D11ShaderResourceView> uv;
    };

    explicit Impl(int inputWidth, int inputHeight)
        : width(inputWidth), height(inputHeight) {}

    ~Impl() {
        ReleaseState();
    }

    auto Upload(ID3D11Texture2D *texture, unsigned arraySlice, ID3D11Device *inputDevice,
                ID3D11DeviceContext *context, HANDLE contextMutex,
                const Yuv420Conversion &conversion,
                void *destination, CUstream_st *stream, CUevent_st *uploadedEvent,
                CUevent_st *preprocessedEvent) -> bool;
    auto ValidateDevice(ID3D11Device *inputDevice) -> bool;
    auto EnsureState(ID3D11Device *inputDevice, const Yuv420Conversion &conversion) -> bool;
    auto CopySlice(ID3D11Texture2D *texture, unsigned arraySlice, ID3D11Device *inputDevice,
                   ID3D11DeviceContext *context, const D3D11_TEXTURE2D_DESC &desc,
                   const Yuv420Conversion &conversion) -> bool;
    auto UploadMapped(const Yuv420Conversion &conversion, void *destination,
                      CUstream_st *stream, CUevent_st *uploadedEvent,
                      CUevent_st *preprocessedEvent) -> bool;
    auto ReleaseSourceCache() -> void;
    auto ReleaseState() -> void;

    int width;
    int height;
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11Buffer> packedBuffer;
    CudaGraphicsResourceHandle packedCudaResource;
    ComPtr<ID3D11UnorderedAccessView> packedUav;
    ComPtr<ID3D11ComputeShader> copyShader;
    ComPtr<ID3D11Buffer> constants;
    UINT packedBufferBytes = 0;
    UINT packedRowPitch = 0;
    UINT packedYPlaneBytes = 0;
    UINT dispatchGroupsX = 0;
    Yuv420Format format = Yuv420Format::NV12;
    D3D11Yuv420FormatInfo formatInfo = GetD3D11Yuv420FormatInfo(Yuv420Format::NV12);
    bool ready = false;
    bool disabled = false;
    ComPtr<ID3D11Texture2D> sourceTexture;
    ComPtr<ID3D11Device> sourceDevice;
    D3D11_TEXTURE2D_DESC sourceDesc {};
    Yuv420Format sourceViewFormat = Yuv420Format::NV12;
    std::vector<SourcePlaneSrvs> sourceSrvs;
    ComPtr<ID3D11Device> validatedDevice;
    bool validatedDeviceCompatible = false;
};

D3D11CudaInput::D3D11CudaInput(int width, int height)
    : _impl(std::make_unique<Impl>(width, height)) {}

D3D11CudaInput::~D3D11CudaInput() = default;

auto D3D11CudaInput::Upload(ID3D11Texture2D *texture, unsigned arraySlice,
                            ID3D11Device *device, ID3D11DeviceContext *context,
                            HANDLE contextMutex, const Yuv420Conversion &conversion, void *destination,
                            CUstream_st *stream, CUevent_st *uploadedEvent,
                            CUevent_st *preprocessedEvent) -> bool {
    return _impl->Upload(texture, arraySlice, device, context, contextMutex, conversion, destination, stream,
                         uploadedEvent, preprocessedEvent);
}

auto D3D11CudaInput::Impl::ReleaseSourceCache() -> void {
    sourceSrvs.clear();
    sourceTexture.Reset();
    sourceDevice.Reset();
    sourceDesc = {};
}

auto D3D11CudaInput::Impl::ReleaseState() -> void {
    ReleaseSourceCache();

    packedCudaResource.Reset();

    constants.Reset();
    copyShader.Reset();
    packedUav.Reset();
    packedBuffer.Reset();
    device.Reset();
    packedBufferBytes = 0;
    packedRowPitch = 0;
    packedYPlaneBytes = 0;
    dispatchGroupsX = 0;
    ready = false;
}

auto D3D11CudaInput::Impl::ValidateDevice(ID3D11Device *inputDevice) -> bool {
    if (inputDevice == nullptr) {
        return false;
    }
    if (validatedDevice.Get() == inputDevice) {
        return validatedDeviceCompatible;
    }

    disabled = false;

    bool compatible = false;
    int currentCudaDevice = -1;
    cudaError_t result = cudaGetDevice(&currentCudaDevice);
    if (result == cudaSuccess) {
        unsigned int count = 0;
        int d3dCudaDevice = -1;
        result = cudaD3D11GetDevices(&count, &d3dCudaDevice, 1, inputDevice,
                                     cudaD3D11DeviceListAll);
        compatible = result == cudaSuccess && count > 0 && d3dCudaDevice == currentCudaDevice;
    }
    if (result != cudaSuccess) {
        cudaGetLastError();
    }

    validatedDevice = inputDevice;
    validatedDeviceCompatible = compatible;
    return compatible;
}

auto D3D11CudaInput::Impl::EnsureState(ID3D11Device *inputDevice,
                                       const Yuv420Conversion &conversion) -> bool {
    if (device.Get() == inputDevice && format == conversion.format && constants != nullptr) {
        return true;
    }

    ReleaseState();
    device = inputDevice;
    format = conversion.format;
    formatInfo = GetD3D11Yuv420FormatInfo(format);

    const size_t rowBytes = static_cast<size_t>(width) * formatInfo.sampleBytes;
    const size_t rowPitch = (rowBytes + sizeof(UINT) - 1) & ~(sizeof(UINT) - 1);
    const size_t yPlaneBytes = rowPitch * height;
    const size_t alignedBytes = yPlaneBytes + rowPitch * (height / 2);

    if (disabled || alignedBytes > (std::numeric_limits<UINT>::max)()) {
        disabled = true;
        ReleaseState();
        return false;
    }

    packedBufferBytes = static_cast<UINT>(alignedBytes);
    packedRowPitch = static_cast<UINT>(rowPitch);
    packedYPlaneBytes = static_cast<UINT>(yPlaneBytes);
    const UINT wordsPerRow = packedRowPitch / sizeof(UINT);
    dispatchGroupsX = (wordsPerRow + 255) / 256;

    bool bufferReady = SUCCEEDED(CreatePackedBuffer(
        device.Get(), packedBufferBytes, packedBuffer.ReleaseAndGetAddressOf()));
    if (bufferReady) {
        bufferReady =
            cudaGraphicsD3D11RegisterResource(
                packedCudaResource.Put(), packedBuffer.Get(),
                cudaGraphicsRegisterFlagsNone) == cudaSuccess &&
            cudaGraphicsResourceSetMapFlags(
                packedCudaResource.Get(), cudaGraphicsMapFlagsReadOnly) == cudaSuccess;
    }

    ready = bufferReady &&
        SUCCEEDED(CreatePackedBufferUav(device.Get(), packedBuffer.Get(), packedBufferBytes,
                                        packedUav.ReleaseAndGetAddressOf())) &&
        SUCCEEDED(CreatePackPlanesShader(device.Get(), format,
                                         copyShader.ReleaseAndGetAddressOf()));

    if (!ready) {
        disabled = true;
        ReleaseState();
        return false;
    }

    const PlaneCopyConstants copyConstants {
        static_cast<UINT>(width),
        static_cast<UINT>(height),
        packedRowPitch,
        packedYPlaneBytes
    };
    D3D11_BUFFER_DESC constantsDesc {};
    constantsDesc.ByteWidth = sizeof(PlaneCopyConstants);
    constantsDesc.Usage = D3D11_USAGE_IMMUTABLE;
    constantsDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA constantsData {};
    constantsData.pSysMem = &copyConstants;
    if (FAILED(device->CreateBuffer(&constantsDesc, &constantsData,
                                    constants.ReleaseAndGetAddressOf()))) {
        ReleaseState();
        return false;
    }

    return true;
}

auto D3D11CudaInput::Impl::CopySlice(ID3D11Texture2D *texture, unsigned arraySlice,
                                     ID3D11Device *inputDevice, ID3D11DeviceContext *context,
                                     const D3D11_TEXTURE2D_DESC &desc,
                                     const Yuv420Conversion &conversion) -> bool {
    if (disabled || !EnsureState(inputDevice, conversion)) {
        return false;
    }

    if (sourceTexture.Get() != texture || sourceViewFormat != conversion.format) {
        ReleaseSourceCache();
        sourceTexture = texture;
        sourceDevice = inputDevice;
        sourceDesc = desc;
        sourceViewFormat = conversion.format;
        sourceSrvs.resize(sourceDesc.ArraySize);
    }

    auto &srvs = sourceSrvs[arraySlice];
    if (srvs.y == nullptr && FAILED(CreateSourceSrv(
            device.Get(), texture, formatInfo.sourceYViewFormat, arraySlice,
            srvs.y.ReleaseAndGetAddressOf()))) {
        return false;
    }
    if (srvs.uv == nullptr && FAILED(CreateSourceSrv(
            device.Get(), texture, formatInfo.sourceUvViewFormat, arraySlice,
            srvs.uv.ReleaseAndGetAddressOf()))) {
        srvs.y.Reset();
        return false;
    }

    D3D11ComputeStateGuard stateGuard(context);
    ID3D11ShaderResourceView *shaderResources[] = { srvs.y.Get(), srvs.uv.Get() };
    context->CSSetShader(copyShader.Get(), nullptr, 0);

    ID3D11Buffer *constantBuffer = constants.Get();
    context->CSSetConstantBuffers(0, 1, &constantBuffer);
    context->CSSetShaderResources(0, 2, shaderResources);

    ID3D11UnorderedAccessView *unorderedAccessView = packedUav.Get();
    context->CSSetUnorderedAccessViews(0, 1, &unorderedAccessView, nullptr);
    context->Dispatch(dispatchGroupsX, static_cast<UINT>(height), 1);

    return true;
}

auto D3D11CudaInput::Impl::UploadMapped(const Yuv420Conversion &conversion, void *destination,
                                        CUstream_st *stream, CUevent_st *uploadedEvent,
                                        CUevent_st *preprocessedEvent) -> bool {
    cudaGraphicsResource_t resource = packedCudaResource.Get();
    if (cudaGraphicsMapResources(1, &resource, stream) != cudaSuccess) {
        disabled = true;
        cudaGetLastError();
        return false;
    }

    void *packedData = nullptr;
    size_t mappedBytes = 0;
    cudaError_t result = cudaGraphicsResourceGetMappedPointer(
        &packedData, &mappedBytes, packedCudaResource.Get());

    bool ok = false;
    if (result == cudaSuccess && mappedBytes >= packedBufferBytes) {
        cudaEventRecord(uploadedEvent, stream);
        result = LaunchYuv420ToFp16Planar(
            packedData, packedRowPitch, packedYPlaneBytes, packedRowPitch,
            destination, width, height, conversion, stream);
        ok = result == cudaSuccess;
        cudaEventRecord(preprocessedEvent, stream);
    }

    if (cudaGraphicsUnmapResources(1, &resource, stream) != cudaSuccess) {
        ok = false;
        cudaGetLastError();
    }

    if (!ok) {
        disabled = true;
        cudaGetLastError();
    }

    return ok;
}

auto D3D11CudaInput::Impl::Upload(ID3D11Texture2D *texture, unsigned arraySlice,
                                  ID3D11Device *inputDevice, ID3D11DeviceContext *context,
                                  HANDLE contextMutex, const Yuv420Conversion &conversion, void *destination,
                                  CUstream_st *stream, CUevent_st *uploadedEvent,
                                  CUevent_st *preprocessedEvent) -> bool {
    if (texture == nullptr || inputDevice == nullptr || context == nullptr || destination == nullptr) {
        return false;
    }

    const bool cachedSource = sourceTexture.Get() == texture;
    D3D11_TEXTURE2D_DESC desc {};
    if (cachedSource) {
        desc = sourceDesc;
    } else {
        texture->GetDesc(&desc);
    }

    const D3D11Yuv420FormatInfo inputFormatInfo = format == conversion.format
        ? formatInfo
        : GetD3D11Yuv420FormatInfo(conversion.format);

    if (desc.Width < static_cast<UINT>(width) || desc.Height < static_cast<UINT>(height) ||
        arraySlice >= desc.ArraySize || desc.Format != inputFormatInfo.sourceFormat ||
        (desc.BindFlags & D3D11_BIND_SHADER_RESOURCE) == 0) {
        return false;
    }

    bool sameDevice = cachedSource && sourceDevice.Get() == inputDevice;
    if (!sameDevice) {
        ComPtr<ID3D11Device> textureDevice;
        texture->GetDevice(textureDevice.ReleaseAndGetAddressOf());
        sameDevice = textureDevice.Get() == inputDevice;
    }

    if (!sameDevice || !ValidateDevice(inputDevice)) {
        return false;
    }

    {
        D3D11ContextGuard guard(contextMutex);
        if (!guard.Locked() ||
            !CopySlice(texture, arraySlice, inputDevice, context, desc, conversion)) {
            return false;
        }
    }

    return UploadMapped(conversion, destination, stream, uploadedEvent, preprocessedEvent);
}

}
