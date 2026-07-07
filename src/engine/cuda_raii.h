// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

namespace MLFilter {

class CudaGraphicsResourceHandle {
public:
    CudaGraphicsResourceHandle() = default;

    ~CudaGraphicsResourceHandle() {
        Reset();
    }

    CudaGraphicsResourceHandle(const CudaGraphicsResourceHandle &) = delete;
    auto operator=(const CudaGraphicsResourceHandle &) -> CudaGraphicsResourceHandle & = delete;

    auto Get() const -> cudaGraphicsResource_t {
        return _resource;
    }

    auto Put() -> cudaGraphicsResource_t * {
        Reset();
        return &_resource;
    }

    auto Reset() -> void {
        if (_resource != nullptr) {
            cudaGraphicsUnregisterResource(_resource);
            _resource = nullptr;
        }
    }

private:
    cudaGraphicsResource_t _resource = nullptr;
};

}
