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
#include "moment_image_device.cuh"
#include "mese/MESE_dynamic.h"
#include "moment_preparation.h"
#include "moment_prediction_coding.h"
#include "coding_transform.cuh"
#include "cut/timing.h"
#include "cut/strided_array.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

using namespace cut;

__constant__ float c_coding_params[512];

int MomentImageDevice::get_best_smem_launch_config(int bytes_per_moment, dim3 &num_blocks,
                                                   dim3 &threads_per_block) const
{
    auto max_moments = view.num_moments;

    int max_shared_mem = g_DeviceProp.sharedMemPerBlock;

    threads_per_block = dim3(8, 4);
    num_blocks.x = (view.width + threads_per_block.x - 1) / threads_per_block.x;
    num_blocks.y = (view.height + threads_per_block.y - 1) / threads_per_block.y;

    int required_smem = threads_per_block.x * threads_per_block.y * max_moments * bytes_per_moment;
    while (required_smem > max_shared_mem)
    {
        threads_per_block.x = threads_per_block.x - 1;
        assert(threads_per_block.x > 0);

        num_blocks.x = (view.width + threads_per_block.x - 1) / threads_per_block.x;
        num_blocks.y = (view.height + threads_per_block.y - 1) / threads_per_block.y;
        required_smem = threads_per_block.x * threads_per_block.y * max_moments * bytes_per_moment;
    }

    return required_smem;
}

__global__ void compaction_copy_kernel(const MomentImageDeviceView mi, const float *in, float *out)
{
    Vec3i gid(__umul24(blockIdx.x, blockDim.x) + threadIdx.x,
              __umul24(blockIdx.y, blockDim.y) + threadIdx.y,
              __umul24(blockIdx.z, blockDim.z) + threadIdx.z);

    if (gid.x >= mi.width || gid.y >= mi.height || gid.z >= mi.num_moments)
        return;

    auto old_idx = mi.get_idx(gid.x, gid.y);
    auto new_idx = mi.index[gid.y * mi.width + gid.x];
    auto num = mi.index[gid.y * mi.width + gid.x + 1] - new_idx;
    if (gid.z >= num)
        return;

    out[new_idx + gid.z] = in[old_idx + gid.z];
}

void MomentImageDevice::compact()
{
    NVTX_RANGE("Image compaction");

    assert(!view.is_compact);
    thrust::device_ptr<uint32_t> index_ptr(index.get());

    auto index_size = view.width * view.height;

    thrust::exclusive_scan(index_ptr, index_ptr + index_size + 1, index_ptr);

    uint32_t new_data_size;
    CHECK_CUDA(cudaMemcpy(&new_data_size, index.get() + index_size, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cut::dev_ptr<float> new_data(new_data_size);

    dim3 threads_per_block(4, 4, 32);
    dim3 num_blocks;
    num_blocks.x = (view.width + threads_per_block.x - 1) / threads_per_block.x;
    num_blocks.y = (view.height + threads_per_block.y - 1) / threads_per_block.y;
    num_blocks.z = (view.num_moments + threads_per_block.z - 1) / threads_per_block.z;
    compaction_copy_kernel<<<num_blocks, threads_per_block>>>(view, data.get(), new_data.get());

    std::swap(data.m_ptr, new_data.m_ptr);
    view.data = data.get();
    data_size = new_data_size;
    view.is_compact = true;
}

__global__ void prediction_encode_kernel(MomentImageDeviceView mi, float bias)
{
    Vec2i gid(__umul24(blockIdx.x, blockDim.x) + threadIdx.x, __umul24(blockIdx.y, blockDim.y) + threadIdx.y);

    if (gid.x >= mi.width || gid.y >= mi.height)
        return;

    auto num_moments = mi.get_num_moments(gid.x, gid.y);

    if (num_moments <= 0)
        return;

    extern __shared__ float_complex storage[];

    int t_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int t_offset = blockDim.x * blockDim.y;
    int array_offset = mi.num_moments * t_offset;

    strided_array<float_complex> code(&storage[t_idx], t_offset);
    strided_array<float_complex> exp_moments(&storage[t_idx + array_offset], t_offset);
    strided_array<float_complex> eval_polynom(&storage[t_idx + 2 * array_offset], t_offset);
    strided_array<float_complex> temp(&storage[t_idx + 3 * array_offset], t_offset);

    auto idx = mi.get_idx(gid.x, gid.y);

    mi.data[idx] = cut::lerp(mi.data[idx], 0.5f, bias);
    for (int i = 1; i < num_moments; ++i)
        mi.data[idx + i] = cut::lerp(mi.data[idx + i], 0.0f, bias);

    trigonometricToExponentialMoments(num_moments, exp_moments, &mi.data[idx]);

    encode(num_moments, exp_moments, code, eval_polynom, temp);

    transform_quantization_real(num_moments, code, mi.data[idx], &mi.data[idx]);
}

__global__ void prediction_encode_warping_kernel(MomentImageDeviceView mi)
{
    Vec3i gid(__umul24(blockIdx.x, blockDim.x) + threadIdx.x,
              __umul24(blockIdx.y, blockDim.y) + threadIdx.y,
              __umul24(blockIdx.z, blockDim.z) + threadIdx.z);

    if (gid.x >= mi.width || gid.y >= mi.height || gid.z >= mi.num_moments)
        return;

    int l = 1 + gid.z;
    auto num_moments = mi.get_num_moments(gid.x, gid.y);
    if (num_moments <= l)
        return;

    auto idx = mi.get_idx(gid.x, gid.y);
    mi.data[idx + l] = prediction_coding_warp(l, mi.data[idx + l], WarpParameters{mi.coding_warp, c_coding_params});
}

void MomentImageDevice::prediction_encode(int coding_warp, float bias)
{
    assert(view.is_compact);

    SCOPED_CUDA_QUERY("Prediction coding");

    view.prediction_code = true;
    view.coding_warp = coding_warp;

    dim3 threads_per_block;
    dim3 num_blocks;
    auto shared_mem = get_best_smem_launch_config(sizeof(float_complex) * 4, num_blocks, threads_per_block);

    prediction_encode_kernel<<<num_blocks, threads_per_block, shared_mem>>>(view, bias);

    if (requires_coding_parameters(coding_warp))
    {
        if (coding_warp == CODING_WARP_DEFAULT_TRANSFORMED)
            coding_params = CodingParamType::quantize(find_coding_transform_parameters_device(view));

        load_coding_params();
    }

    threads_per_block = dim3(4, 4, 16);
    num_blocks.x = (view.width + threads_per_block.x - 1) / threads_per_block.x;
    num_blocks.y = (view.height + threads_per_block.y - 1) / threads_per_block.y;
    num_blocks.z = (view.num_moments - 1 + threads_per_block.z - 1)  / threads_per_block.z;

    prediction_encode_warping_kernel<<<num_blocks, threads_per_block>>>(view);
}

__global__ void revert_prediction_coding_kernel(MomentImageDeviceView mi)
{
    Vec2i gid(__umul24(blockIdx.x, blockDim.x) + threadIdx.x, __umul24(blockIdx.y, blockDim.y) + threadIdx.y);

    if (gid.x >= mi.width || gid.y >= mi.height)
        return;

    auto num_moments = mi.get_num_moments(gid.x, gid.y);

    if (num_moments <= 0)
        return;

    extern __shared__ float_complex storage[];

    int t_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int t_offset = blockDim.x * blockDim.y;
    int array_offset = mi.num_moments * t_offset;

    strided_array<float_complex> code(&storage[t_idx], t_offset);
    strided_array<float_complex> exp_moments(&storage[t_idx + array_offset], t_offset);
    strided_array<float_complex> eval_polynom(&storage[t_idx + 2 * array_offset], t_offset);
    strided_array<float_complex> temp(&storage[t_idx + 3 * array_offset], t_offset);

    auto idx = mi.get_idx(gid.x, gid.y);

    transform_dequantization_real(num_moments, &mi.data[idx], code, WarpParameters{mi.coding_warp, c_coding_params});

    decode(num_moments, code, exp_moments, eval_polynom, temp);

    exponentialToTrigonometricMoments(num_moments, temp, exp_moments);

    for (int l = 0; l < num_moments; ++l)
        mi.data[idx + l] = temp[l].x;
}

void MomentImageDevice::revert_prediction_coding()
{
    SCOPED_CUDA_QUERY("Revert prediction coding");
    assert(view.prediction_code);

    load_coding_params();

    dim3 threads_per_block;
    dim3 num_blocks;
    auto shared_mem = get_best_smem_launch_config(sizeof(float_complex) * 4, num_blocks, threads_per_block);

    revert_prediction_coding_kernel<<<num_blocks, threads_per_block, shared_mem>>>(view);

    view.prediction_code = false;
}

__global__ void prepareMomentsFromPredictionCodingKernel(MomentImageDeviceView mi, float *pmoments)
{
    Vec2i gid(__umul24(blockIdx.x, blockDim.x) + threadIdx.x, __umul24(blockIdx.y, blockDim.y) + threadIdx.y);
    if (gid.x >= mi.width || gid.y >= mi.height)
        return;

    Vec2f ndc(gid.x / static_cast<float>(mi.width) * 2.f - 1.f, gid.y / static_cast<float>(mi.height) * 2.f - 1.f);

    auto idx = mi.get_idx(gid.x, gid.y);
    auto num_moments = mi.get_num_moments(gid.x, gid.y);

    if (num_moments <= 0)
        return;

    float *prepared_moments = &pmoments[idx];

    extern __shared__ float_complex storage[];

    int t_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int t_offset = blockDim.x * blockDim.y;
    int array_offset = mi.num_moments * t_offset;

    strided_array<float_complex> code(&storage[t_idx], t_offset);
    strided_array<float_complex> exp_moments(&storage[t_idx + array_offset], t_offset);
    strided_array<float_complex> eval_polynom(&storage[t_idx + 2 * array_offset], t_offset);
    strided_array<float_complex> temp(&storage[t_idx + 3 * array_offset], t_offset);

    prepare_moments_from_pred_coding(num_moments, &mi.data[idx], prepared_moments, code, exp_moments, eval_polynom,
                                     temp, WarpParameters{mi.coding_warp, c_coding_params});
}

__global__ void prepareMomentsKernel(MomentImageDeviceView mi, float *pmoments, float bias)
{
    Vec2i gid(__umul24(blockIdx.x, blockDim.x) + threadIdx.x, __umul24(blockIdx.y, blockDim.y) + threadIdx.y);
    if (gid.x >= mi.width || gid.y >= mi.height)
        return;

    Vec2f ndc(gid.x / static_cast<float>(mi.width) * 2.f - 1.f, gid.y / static_cast<float>(mi.height) * 2.f - 1.f);

    auto idx = mi.get_idx(gid.x, gid.y);
    auto num_moments = mi.get_num_moments(gid.x, gid.y);

    if (num_moments == 0)
        return;

    extern __shared__ float_complex storage[];

    int t_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int t_offset = blockDim.x * blockDim.y;
    int array_offset = mi.num_moments * t_offset;

    strided_array<float_complex> temp0(&storage[t_idx], t_offset);
    strided_array<float_complex> temp1(&storage[t_idx + array_offset], t_offset);
    strided_array<float_complex> temp2(&storage[t_idx + 2 * array_offset], t_offset);

    prepare_moments_from_std_coding(num_moments, &mi.data[idx], &pmoments[idx], temp0, temp1, temp2, bias);
}

void MomentImageDevice::prepare_moments_device(float bias)
{
    dim3 threadsPerBlock(4, 4);
    dim3 numBlocks;
    numBlocks.x = (view.width + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (view.height + threadsPerBlock.y - 1) / threadsPerBlock.y;

    cut::dev_ptr<float> pmoments(data_size);
    if (view.prediction_code)
    {
        SCOPED_CUDA_QUERY("Prediction preparation");

        load_coding_params();

        auto shared_mem = get_best_smem_launch_config(sizeof(float_complex) * 4, numBlocks, threadsPerBlock);

        prepareMomentsFromPredictionCodingKernel<<<numBlocks, threadsPerBlock, shared_mem>>>(view, pmoments.get());
    }
    else
    {
        SCOPED_CUDA_QUERY("Standard preparation");

        auto shared_mem = get_best_smem_launch_config(sizeof(float_complex) * 3, numBlocks, threadsPerBlock);

        prepareMomentsKernel<<<numBlocks, threadsPerBlock, shared_mem>>>(view, pmoments.get(), bias);
    }

    std::swap(data, pmoments);
    view.data = data.get();
}

void MomentImageDevice::load_coding_params()
{
    if (requires_coding_parameters(view.coding_warp))
    {
        assert(!coding_params.empty());
        auto params = CodingParamType::dequantize(coding_params);
        CHECK_CUDA(cudaMemcpyToSymbol(c_coding_params, params.data(), sizeof(float) * params.size(), 0,
                                      cudaMemcpyHostToDevice));
    }
}
