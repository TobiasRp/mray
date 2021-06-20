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
#include "coding_transform.cuh"
#include "cut/timing.h"
#include "cut/cuda_math.h"
using namespace cut;

#include <cooperative_groups.h>
using namespace cooperative_groups;

__device__ float2 reduce_minmax_tile(thread_block_tile<32> g, float val)
{
    float2 v = make_float2(val, val);
    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        v.x = cut::min(v.x, g.shfl_down(v.x, i));
        v.y = cut::max(v.y, g.shfl_down(v.y, i));
    }
    return v; // only thread 0 returns min, max!
}

__global__ void find_ranges_kernel(const MomentImageDeviceView mi, uint32_t *params)
{
    Vec3i gid(__umul24(blockIdx.x, blockDim.x) + threadIdx.x,
              __umul24(blockIdx.y, blockDim.y) + threadIdx.y,
              __umul24(blockIdx.z, blockDim.z) + threadIdx.z);

    if (gid.x >= mi.width || gid.y >= mi.height || gid.z >= mi.num_moments - 1)
        return;

    auto l = 1 + gid.z;
    auto num_moments = mi.get_num_moments(gid.x, gid.y);
    if (num_moments <= l)
        return;

    auto idx = mi.get_idx(gid.x, gid.y);
    auto value = cut::clamp(mi.data[idx + l], -1.0f, 1.0f);

    __shared__ uint32_t s_min_value;
    __shared__ uint32_t s_max_value;

    auto block = this_thread_block();
    if (block.thread_rank() == 0)
    {
        s_min_value = 0;
        s_max_value = 0;
    }
    block.sync();

    auto tile = tiled_partition<32>(block);
    auto minmax = reduce_minmax_tile(tile, value);
    if (tile.thread_rank() == 0)
    {
        uint32_t min = cut::abs(minmax.x) * UINT32_MAX;
        uint32_t max = cut::abs(minmax.y) * UINT32_MAX;
        atomicMax(&s_min_value, min);
        atomicMax(&s_max_value, max);
    }

    block.sync();
    if (block.thread_rank() == 0)
    {
        atomicMax(&params[2 * gid.z], s_min_value);
        atomicMax(&params[2 * gid.z + 1], s_max_value);
    }
}

__global__ void zero_params_kernel(int num, uint32_t *params)
{
    int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (id >= num)
        return;
    params[id] = 0;
}

__global__ void uint_to_float_kernel(int num, float *params)
{
    int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (id >= num)
        return;
    params[id] = static_cast<float>(reinterpret_cast<uint32_t*>(params)[id]) / UINT32_MAX;
}

vector<float> find_coding_transform_parameters_device(const MomentImageDeviceView &img)
{
    NVTX_RANGE("Find coding parameters");

    int num_dists = img.num_moments - 1;

    dev_ptr<float> d_params(num_dists * 2);
    zero_params_kernel<<<1, num_dists*2>>>(num_dists * 2, reinterpret_cast<uint32_t *>(d_params.get()));

    dim3 threads_per_block(32, 32, 1);
    dim3 num_blocks;
    num_blocks.x = (img.width + threads_per_block.x - 1) / threads_per_block.x;
    num_blocks.y = (img.height + threads_per_block.y - 1) / threads_per_block.y;
    num_blocks.z = (num_dists + threads_per_block.z - 1)  / threads_per_block.z;

    find_ranges_kernel<<<num_blocks, threads_per_block>>>(img, reinterpret_cast<uint32_t *>(d_params.get()));

    uint_to_float_kernel<<<1, num_dists*2>>>(num_dists * 2, d_params.get());

    vector<float> params(num_dists * 2);
    d_params.copyToHost(params.data(), num_dists * 2);

    for (int pidx = 0; pidx < num_dists; ++pidx)
    {
        params[pidx * 2 + 1] = cut::abs(params[pidx * 2 + 1] + params[pidx * 2]) * 0.5f;

        assert(params[pidx * 2 + 1] >= 0.0f && params[pidx * 2 + 1] <= 1.0f);
        assert(params[pidx * 2] >= 0.0f && params[pidx * 2] <= 1.0f);
    }

    return params;
}