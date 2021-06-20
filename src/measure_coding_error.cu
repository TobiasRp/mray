#include "measure_coding_error.cuh"
#include "moment_image_device.cuh"
#include "moment_prediction_coding.h"
#include "mese/MESE_dynamic.h"
#include "cut/strided_array.h"

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

using namespace cut;

__constant__ float c_coding_params[512];
__constant__ Byte c_table[255];

void setup_coding_error_device(const MomentImageDevice &d_img)
{
    auto params = CodingParamType::dequantize(d_img.coding_params);
    CHECK_CUDA(
        cudaMemcpyToSymbol(c_coding_params, params.data(), sizeof(float) * params.size(), 0, cudaMemcpyHostToDevice));
}

__device__ __forceinline__ float apply_quantization(float value, float b)
{
    if (b == 32)
        return value;

    assert(value >= 0.0f && value <= 1.0f); // Holds for prediction coding
    float pow = std::pow(2.0f, b);
    float q = std::floor(value * pow);
    return q / pow;
}

__global__ void apply_quantization_kernel(MomentImageDeviceView mi, int b_idx, float b)
{
    Vec2i gid(__umul24(blockIdx.x, blockDim.x) + threadIdx.x, __umul24(blockIdx.y, blockDim.y) + threadIdx.y);
    if (gid.x >= mi.width || gid.y >= mi.height)
        return;

    if (mi.get_num_moments(gid.x, gid.y) <= b_idx)
        return;

    auto idx = mi.get_idx(gid.x, gid.y);

    auto m_b = mi.data[idx + b_idx];
    mi.data[idx + b_idx] = apply_quantization(m_b, b);
}

__global__ void apply_quantization_kernel(MomentImageDeviceView mi, const float *orig_data, int b_idx, float b)
{
    Vec2i gid(__umul24(blockIdx.x, blockDim.x) + threadIdx.x, __umul24(blockIdx.y, blockDim.y) + threadIdx.y);
    if (gid.x >= mi.width || gid.y >= mi.height)
        return;

    if (mi.get_num_moments(gid.x, gid.y) <= b_idx)
        return;

    auto idx = mi.get_idx(gid.x, gid.y);
    mi.data[idx + b_idx] = apply_quantization(orig_data[idx + b_idx], b);
}

__global__ void apply_quantization_kernel(MomentImageDeviceView mi)
{
    Vec2i gid(__umul24(blockIdx.x, blockDim.x) + threadIdx.x, __umul24(blockIdx.y, blockDim.y) + threadIdx.y);
    if (gid.x >= mi.width || gid.y >= mi.height)
        return;

    auto num_moments = mi.get_num_moments(gid.x, gid.y);
    if (num_moments == 0)
        return;

    auto idx = mi.get_idx(gid.x, gid.y);
    for (int l = 0; l < num_moments; ++l)
    {
        auto m_b = mi.data[idx + l];
        mi.data[idx + l] = apply_quantization(m_b, c_table[l]);
    }
}

void apply_quantization_device(const MomentImageDevice &d_img, int b_idx, float b)
{
    dim3 threads_per_block(8, 8);
    dim3 num_blocks;
    num_blocks.x = (d_img.view.width + threads_per_block.x - 1) / threads_per_block.x;
    num_blocks.y = (d_img.view.height + threads_per_block.y - 1) / threads_per_block.y;

    apply_quantization_kernel<<<num_blocks, threads_per_block>>>(d_img.view, b_idx, b);
}

void apply_quantization_device(const MomentImageDevice &d_img, const float *d_orig_data, int b_idx, float b)
{
    dim3 threads_per_block(8, 8);
    dim3 num_blocks;
    num_blocks.x = (d_img.view.width + threads_per_block.x - 1) / threads_per_block.x;
    num_blocks.y = (d_img.view.height + threads_per_block.y - 1) / threads_per_block.y;

    apply_quantization_kernel<<<num_blocks, threads_per_block>>>(d_img.view, d_orig_data, b_idx, b);
}

void apply_quantization_device(const MomentImageDevice &d_img, const vector<Byte> &table)
{
    dim3 threads_per_block(8, 8);
    dim3 num_blocks;
    num_blocks.x = (d_img.view.width + threads_per_block.x - 1) / threads_per_block.x;
    num_blocks.y = (d_img.view.height + threads_per_block.y - 1) / threads_per_block.y;

    CHECK_CUDA(cudaMemcpyToSymbol(c_table, table.data(), table.size()));

    apply_quantization_kernel<<<num_blocks, threads_per_block>>>(d_img.view);
}

__global__ void revert_quantization_device(MomentImageDeviceView mi, int b_idx, const float *orig_data)
{
    Vec2i gid(__umul24(blockIdx.x, blockDim.x) + threadIdx.x, __umul24(blockIdx.y, blockDim.y) + threadIdx.y);
    if (gid.x >= mi.width || gid.y >= mi.height)
        return;

    if (mi.get_num_moments(gid.x, gid.y) <= b_idx)
        return;

    auto idx = mi.get_idx(gid.x, gid.y);
    mi.data[idx + b_idx] = orig_data[idx + b_idx];
}

void revert_quantization_device(const MomentImageDevice &d_img, int b_idx, const float *d_orig_data)
{
    dim3 threads_per_block(8, 8);
    dim3 num_blocks;
    num_blocks.x = (d_img.view.width + threads_per_block.x - 1) / threads_per_block.x;
    num_blocks.y = (d_img.view.height + threads_per_block.y - 1) / threads_per_block.y;

    revert_quantization_device<<<num_blocks, threads_per_block>>>(d_img.view, b_idx, d_orig_data);
}

__global__ void mark_nonzero_kernel(MomentImageDeviceView mi, uint32_t *indices)
{
    Vec2i gid(__umul24(blockIdx.x, blockDim.x) + threadIdx.x, __umul24(blockIdx.y, blockDim.y) + threadIdx.y);
    if (gid.x >= mi.width || gid.y >= mi.height)
        return;

    auto num_moments = mi.get_num_moments(gid.x, gid.y);

    if (num_moments <= 0)
        indices[gid.y * mi.width + gid.x] = 0;
    else
        indices[gid.y * mi.width + gid.x] = 1;
}

__global__ void compute_reference_kernel(MomentImageDeviceView mi, float *ref_moments)
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
        ref_moments[idx + l] = temp[l].x;
}

uint32_t compute_reference_device(const MomentImageDevice &d_img, float *d_moments, uint32_t *d_indices)
{
    dim3 threads_per_block(8, 8);
    dim3 num_blocks;
    num_blocks.x = (d_img.view.width + threads_per_block.x - 1) / threads_per_block.x;
    num_blocks.y = (d_img.view.height + threads_per_block.y - 1) / threads_per_block.y;

    mark_nonzero_kernel<<<num_blocks, threads_per_block>>>(d_img.view, d_indices);

    thrust::device_ptr<uint32_t> ind(d_indices);
    thrust::exclusive_scan(ind, ind + d_img.view.width * d_img.view.height, ind);

    uint32_t size;
    CHECK_CUDA(cudaMemcpy(&size, &d_indices[d_img.view.width * d_img.view.height - 1], sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    auto shared_mem = d_img.get_best_smem_launch_config(sizeof(float_complex) * 4, num_blocks, threads_per_block);
    compute_reference_kernel<<<num_blocks, threads_per_block, shared_mem>>>(d_img.view, d_moments);

    return size;
}

__global__ void compute_coding_error_kernel(MomentImageDeviceView mi, const float *ref_moments,
                                            const uint32_t *indices, float *errors)
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

    auto sum = 0.0f;
    for (int l = 0; l < num_moments; ++l)
    {
        auto diff = ref_moments[idx+l] - temp[l].x;
        sum += diff * diff;
    }

    auto rmse = sqrtf(sum);
    // rRMSE: Divide by average

    auto index = indices[gid.y * mi.width + gid.x];
    errors[index] = rmse / ref_moments[idx];
}

void compute_coding_error_device(const MomentImageDevice &d_img, const float *d_ref_moments,
                                 float *d_errors, const uint32_t *d_indices, size_t size, float &error)
{
    dim3 threads_per_block;
    dim3 num_blocks;
    auto shared_mem = d_img.get_best_smem_launch_config(sizeof(float_complex) * 4, num_blocks, threads_per_block);

    compute_coding_error_kernel<<<num_blocks, threads_per_block, shared_mem>>>(d_img.view, d_ref_moments, d_indices,
                                                                               d_errors);

    thrust::device_ptr<float> d_exp(d_errors);
    float sum = thrust::reduce(d_exp, d_exp + size, (float)0.f, thrust::plus<float>());
    error = sum / static_cast<float>(size);
}