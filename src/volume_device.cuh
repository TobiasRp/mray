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
#ifndef MRAY_VOLUME_DEVICE_CUH
#define MRAY_VOLUME_DEVICE_CUH

#include "volume.h"
#include "particle_interpolate.h"
#include "scatter_grid_device.cuh"

struct RegularGridDeviceView
{
    Vec3i size;
    Range<Vec3f> domain;
    cudaTextureObject_t tex;

    __device__ inline float sample(Vec3f p) const
    {
        Vec3f n = cut::div(p - domain.min, domain.max - domain.min);
        return tex3D<float>(tex, n.x, n.y, n.z);
    }

    __device__ inline float get_step_size() const
    {
        auto cells = cut::div(domain.max - domain.min, cut::make_vec3f(size));
        return cut::min3(cells.x, cells.y, cells.z);
    }
};

struct RegularGridDevice
{
    RegularGridDeviceView view;

    cudaArray_t data;

    RegularGridDevice(const RegularGrid &grid)
    {
        view.size = grid.size;
        view.domain = grid.domain;

        int byte_size = grid.get_byte_size();

        cudaExtent extent{static_cast<size_t>(grid.size.x), static_cast<size_t>(grid.size.y),
                          static_cast<size_t>(grid.size.z)};
        cudaChannelFormatDesc formatDesc{
            byte_size * 8, 0, 0, 0, grid.is_normalized() ? cudaChannelFormatKindUnsigned : cudaChannelFormatKindFloat};

        CHECK_CUDA(cudaMalloc3DArray(&data, &formatDesc, extent));
        auto srcPtr = make_cudaPitchedPtr((void *)grid.data.data(), byte_size * grid.size.x, grid.size.x, grid.size.y);

        cudaMemcpy3DParms copyParams;
        memset(&copyParams, 0, sizeof(copyParams));
        copyParams.dstArray = data;
        copyParams.srcPtr = srcPtr;
        copyParams.extent = cudaExtent{static_cast<size_t>(grid.size.x), static_cast<size_t>(grid.size.y),
                                       static_cast<size_t>(grid.size.z)};
        copyParams.kind = cudaMemcpyHostToDevice;
        CHECK_CUDA(cudaMemcpy3D(&copyParams));

        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = data;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = grid.is_normalized() ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.addressMode[2] = cudaAddressModeClamp;
        texDesc.normalizedCoords = 1;

        CHECK_CUDA(cudaCreateTextureObject(&view.tex, &resDesc, &texDesc, NULL));
    }

    RegularGridDevice(ScatterGridDevice grid)
    {
        view.size = grid.size;
        view.domain = grid.domain;
        data = grid.data;
        grid.data = nullptr;

        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = data;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.addressMode[2] = cudaAddressModeClamp;
        texDesc.normalizedCoords = 1;

        CHECK_CUDA(cudaCreateTextureObject(&view.tex, &resDesc, &texDesc, NULL));
    }

    RegularGridDevice(const RegularGridDevice &) = delete;
    RegularGridDevice &operator=(const RegularGridDevice &) = delete;

    RegularGridDevice(RegularGridDevice &&) = default;
    RegularGridDevice &operator=(RegularGridDevice &&) = default;

    ~RegularGridDevice()
    {
        CHECK_CUDA(cudaDestroyTextureObject(view.tex));
        CHECK_CUDA(cudaFreeArray(data));
    }
};

struct ParticlesDeviceView
{
    Range<Vec3f> domain;
    uint32_t size;

    float smoothing_length;
    const float *x, *y, *z, *values;
    UniformGridDevice grid;

    __device__ float sample(Vec3f p) const
    {
        InterpolateParticlesOp<true> op;
        op.p = p;
        op.values = values;
        op.inv_h = 1.f / smoothing_length;

        grid.forEachNeighbor<true>(p, size, x, y, z, op);

        op.normalize();
        return op.value;
    }

    __device__ float get_step_size() const { return smoothing_length * 0.5f; }
};

struct ParticlesDevice
{
    ParticlesDeviceView view;

    cut::dev_ptr<float> x;
    cut::dev_ptr<float> y;
    cut::dev_ptr<float> z;
    cut::dev_ptr<float> values;

    UniformGridGPU grid;

    ParticlesDevice(const Particles &particles)
        : x(particles.x)
        , y(particles.y)
        , z(particles.z)
        , values(particles.values)
        , grid(*particles.grid.get())
    {
        view.domain = particles.domain;
        view.size = particles.size();
        view.smoothing_length = particles.smoothing_length;
        view.x = x.get();
        view.y = y.get();
        view.z = z.get();
        view.values = values.get();
        view.grid = grid.d_grid;
    }
};

#endif // MRAY_VOLUME_DEVICE_CUH
