// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include <windows.h>

#include "../color/yuv420_format.h"

struct ID3D11Device;
struct ID3D11DeviceContext;
struct ID3D11Texture2D;
struct CUevent_st;
struct CUstream_st;

namespace MLFilter {

// Stages native D3D11 NV12/P010 decoder textures into a CUDA input tensor.
// This NVIDIA-specific implementation is kept separate from inference orchestration so another
// GPU backend can provide its own native-input path without changing InferenceSession.
class D3D11CudaInput {
public:
    D3D11CudaInput(int width, int height);
    ~D3D11CudaInput();

    D3D11CudaInput(const D3D11CudaInput &) = delete;
    auto operator=(const D3D11CudaInput &) -> D3D11CudaInput & = delete;

    auto Upload(ID3D11Texture2D *texture,
                unsigned arraySlice,
                ID3D11Device *device,
                ID3D11DeviceContext *context,
                HANDLE contextMutex,
                const Yuv420Conversion &conversion,
                void *destination,
                CUstream_st *stream
#ifdef MLFILTER_ENABLE_STAGE_TIMINGS
                ,
                CUevent_st *uploadedEvent,
                CUevent_st *preprocessedEvent
#endif
                ) -> bool;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

}
