// Copyright (c) 2020, Tobias Rapp
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the Karlsruhe Institute of Technology nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#ifndef MRAY_SCATTER_GRID_DEVICE_CUH
#define MRAY_SCATTER_GRID_DEVICE_CUH

#include "common.h"

struct ScatterGridDevice
{
    Vec3i size;
    Range<Vec3f> domain;

    cudaArray_t data;
    cudaSurfaceObject_t surface;

    ScatterGridDevice(Vec3i size, Range<Vec3f> domain)
        : size(size)
        , domain(domain)
    {
        cudaExtent extent{static_cast<size_t>(size.x), static_cast<size_t>(size.y), static_cast<size_t>(size.z)};
        cudaChannelFormatDesc formatDesc{32, 0, 0, 0, cudaChannelFormatKindFloat};

        CHECK_CUDA(cudaMalloc3DArray(&data, &formatDesc, extent, cudaArraySurfaceLoadStore));

        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = data;

        CHECK_CUDA(cudaCreateSurfaceObject(&surface, &resDesc));
    }

    ~ScatterGridDevice()
    {
        CHECK_CUDA(cudaDestroySurfaceObject(surface));

        // FIXME: Doesn't frees data since it is (currently!) used read-only later on
        // This is obviously not correct in general!!!!
        //        CHECK_CUDA(cudaFreeArray(data));
    }

    __device__ Vec3i get_voxel(Vec3f p) const
    {
        Vec3f np = cut::div(p - domain.min, domain.max - domain.min);
        return cut::cast<float, int>(cut::mul(np, cut::cast<int, float>(size)));
    }

    __device__ void write(Vec3i c, float T) { surf3Dwrite<float>(T, surface, c.x * sizeof(float), c.y, c.z); }

    __device__ void scatter(Vec3f pt, float T)
    {
        Vec3f np = cut::div(pt - domain.min, domain.max - domain.min);
        Vec3i ip = cut::cast<float, int>(cut::mul(np, cut::cast<int, float>(size)));
        if (ip.x >= 0 && ip.x < size.x && ip.y >= 0 && ip.y < size.y && ip.z >= 0 && ip.z < size.z)
            write(ip, T);
    }
};

#endif // MRAY_SCATTER_GRID_DEVICE_CUH
