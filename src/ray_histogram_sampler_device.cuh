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
#ifndef MRAY_RAYHISTOGRAM_SAMPLER_DEVICE_H
#define MRAY_RAYHISTOGRAM_SAMPLER_DEVICE_H

#include <curand.h>
#include <curand_kernel.h>

#include "ray_histogram_sampler.h"

// Without importance sampling (Sec. 5.1)
struct SimpleRayHistogramSamplerDevice
{
    curandState *state;
    int width, height;

    void init(int w, int h)
    {
        width = w;
        height = h;
        state = CUDA_MALLOC_TYPED(curandState, w * h);
    }

    void destroy() { CUDA_FREE(state); }

    /** Setups the random number generator state **/
    __device__ void setup(int x, int y)
    {
        auto idx = y * width + x;
        curand_init(0 /*seed*/, idx /*sequence*/, 0, &state[idx]);
    }

    __device__ int getIndex() const
    {
        auto idx =
            (__umul24(blockIdx.x, blockDim.x) + threadIdx.x) + (__umul24(blockIdx.y, blockDim.y) + threadIdx.y) * width;
        return idx % (width * height);
    }

    __device__ float sample()
    {
        auto id = getIndex();
        return curand_uniform(&state[id]);
    }

    __device__ float operator()(const uint16_t *bins, const uint16_t *f, int size, float f_sum)
    {
        return sample_without_importance(sample(), bins, f, size, f_sum);
    }
};

// Importance sampling (Sec. 5.1)
struct ImportanceRayHistogramSamplerDevice
{
    curandState *state;
    int width, height;

    float *d_importance;

    void init(int w, int h, const TransferFunction &tf)
    {
        width = w;
        height = h;
        state = CUDA_MALLOC_TYPED(curandState, w * h);

        auto importance = compute_importance(tf);
        d_importance = CUDA_MALLOC_TYPED(float, importance.size());
        CHECK_CUDA(
            cudaMemcpy(d_importance, importance.data(), importance.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    void destroy()
    {
        CUDA_FREE(state);
        CUDA_FREE(d_importance);
    }

    /** Setups the random number generator state **/
    __device__ void setup(int x, int y)
    {
        auto idx = y * width + x;
        curand_init(0 /*seed*/, idx /*sequence*/, 0, &state[idx]);
    }

    __device__ int getIndex() const
    {
        auto idx =
            (__umul24(blockIdx.x, blockDim.x) + threadIdx.x) + (__umul24(blockIdx.y, blockDim.y) + threadIdx.y) * width;
        return idx % (width * height);
    }

    __device__ float sample()
    {
        auto id = getIndex();
        return curand_uniform(&state[id]);
    }

    __device__ float operator()(const uint16_t *bins, const uint16_t *f, int size, float f_sum)
    {
        return sample_importance(sample(), bins, f, size, f_sum, d_importance);
    }
};

#endif // MRAY_RAYHISTOGRAM_SAMPLER_DEVICE_H
