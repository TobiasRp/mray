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
#include "moment_raymarcher.cuh"
#include "raymarch_generate.h"
#include "raymarch_reconstruct.h"
#include "raymarch_reference.h"
#include "raymarch_generate_samples.h"
#include "raymarch_rayhistogram.h"
#include "raymarch_single_scattering_cache.h"
#include "raymarch_resample.h"
#include "moment_compaction.h"

#include "cut/raytracing.h"
#include "cut/timing.h"
#include "cut/strided_array.h"

#include "moment_image_device.cuh"
#include "transfer_function_device.cuh"
#include "volume_device.cuh"
#include "color_image_device.cuh"
#include "samples_image_device.cuh"
#include "ray_histogram_image_device.cuh"
#include "ray_histogram_sampler_device.cuh"
#include "single_scattering_image_device.cuh"
#include "single_scattering_cache_device.cuh"
#include "scatter_grid_device.cuh"

__constant__ cut::Camera c_camera;
__constant__ GenerationParameters c_gen_params;
__constant__ ReconstructionParameters c_rec_params;
__constant__ ErrorReconstructionParameters c_error_params;
__constant__ ReferenceParameters c_ref_params;
__constant__ RayHistogramReconstructionParameters c_rayhist_rec_params;

using namespace cut;

__device__ __forceinline__ Vec2i pixel_idx()
{
    return Vec2i(__umul24(blockIdx.x, blockDim.x) + threadIdx.x, __umul24(blockIdx.y, blockDim.y) + threadIdx.y);
}

template <typename Volume, typename SingleScattering>
__global__ void generate_reference_kernel(TransferFunctionDevice tf, Volume volume, SingleScattering ss,
                                          ColorImageDeviceView img)
{
    auto gid = pixel_idx();
    if (gid.x >= img.width || gid.y >= img.height)
        return;

    Vec2f ndc(gid.x / static_cast<float>(img.width) * 2.f - 1.f, gid.y / static_cast<float>(img.height) * 2.f - 1.f);

    img.set_color(gid.x, gid.y,
                  reference_ray(volume, get_eye_ray(c_camera, ndc), c_camera.znear, tf, ss, c_ref_params));
}

template <typename VolumeDevice, typename SingleScattering>
void generate_reference_device_helper(cut::Camera cam, const VolumeDevice &volume, const TransferFunction &tf,
                                      ColorImageHost &img, const SingleScattering &ss,
                                      const ReferenceParameters &params)
{
    ColorImageDevice cimg_mgr(img);
    TransferFunctionDevice d_tf(tf);

    CHECK_CUDA(cudaMemcpyToSymbol(c_camera, &cam, sizeof(cut::Camera), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(c_ref_params, &params, sizeof(ReferenceParameters), 0, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks;
    numBlocks.x = (img.width + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (img.height + threadsPerBlock.y - 1) / threadsPerBlock.y;

    {
        SCOPED_CUDA_QUERY("Raymarching");
        generate_reference_kernel<<<numBlocks, threadsPerBlock>>>(d_tf, volume.view, ss, cimg_mgr.view);
    }

    cimg_mgr.copy_back(img);
}

void generate_reference_device(cut::Camera cam, const Particles &volume, const TransferFunction &tf,
                               ColorImageHost &img, const ReferenceParameters &params)
{
    ParticlesDevice d_part(volume);
    generate_reference_device_helper(cam, d_part, tf, img, SingleScatteringBruteForce{}, params);
}

void generate_reference_device(cut::Camera cam, const RegularGrid &volume, const TransferFunction &tf,
                               ColorImageHost &img, const ReferenceParameters &params)
{
    RegularGridDevice d_grid(volume);
    generate_reference_device_helper(cam, d_grid, tf, img, SingleScatteringBruteForce{}, params);
}

template <typename Volume> __global__ void generate_samples_kernel(Volume volume, SamplesImageDeviceView img)
{
    auto gid = pixel_idx();
    if (gid.x >= img.width || gid.y >= img.height)
        return;

    Vec2f ndc(gid.x / static_cast<float>(img.width) * 2.f - 1.f, gid.y / static_cast<float>(img.height) * 2.f - 1.f);

    auto idx = img.get_idx(gid.x, gid.y);

    generate_samples_ray(volume, get_eye_ray(c_camera, ndc), c_camera.znear, SamplesWriter(&img.data[idx]),
                         c_gen_params);
}

template <typename VolumeDevice>
void generate_samples_device_helper(cut::Camera cam, const VolumeDevice &d_volume, SamplesImageHost &img,
                                    const GenerationParameters &params)
{
    CHECK_CUDA(cudaMemcpyToSymbol(c_camera, &cam, sizeof(cut::Camera), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(c_gen_params, &params, sizeof(GenerationParameters), 0, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks;
    numBlocks.x = (img.width + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (img.height + threadsPerBlock.y - 1) / threadsPerBlock.y;

    SamplesImageDevice d_img(img);

    generate_samples_kernel<<<numBlocks, threadsPerBlock>>>(d_volume.view, d_img.view);

    d_img.copy_back(img);
}

void generate_samples_device(cut::Camera cam, const Particles &volume, SamplesImageHost &img,
                             const GenerationParameters &params)
{
    ParticlesDevice d_part(volume);
    generate_samples_device_helper(cam, d_part, img, params);
}

void generate_samples_device(cut::Camera cam, const RegularGrid &volume, SamplesImageHost &img,
                             const GenerationParameters &params)
{
    RegularGridDevice d_grid(volume);
    generate_samples_device_helper(cam, d_grid, img, params);
}

template <typename Volume> __global__ void generate_moments_kernel(Volume volume, MomentImageDeviceView img)
{
    auto gid = pixel_idx();
    if (gid.x >= img.width || gid.y >= img.height)
        return;

    Vec2f ndc(gid.x / static_cast<float>(img.width) * 2.f - 1.f, gid.y / static_cast<float>(img.height) * 2.f - 1.f);
    auto idx = img.get_idx(gid.x, gid.y);

    extern __shared__ float fast_storage[];
    int t_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int t_offset = blockDim.x * blockDim.y;

    strided_array<float> trig_moments(&fast_storage[t_idx], t_offset);
    for (int m = 0; m < img.num_moments; ++m)
        trig_moments[m] = 0.0f;

    auto b = generate_ray(volume, get_eye_ray(c_camera, ndc), c_camera.znear, trig_moments, &img.data[idx],
                          img.num_moments, c_gen_params);

    if (c_gen_params.transform_to_bounds)
        img.bounds[gid.y * img.width + gid.x] = b;
}

__global__ void compute_num_moments_kernel(MomentImageDeviceView img)
{
    auto gid = pixel_idx();
    if (gid.x >= img.width || gid.y >= img.height)
        return;

    extern __shared__ float_complex storage[];

    int t_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int t_offset = blockDim.x * blockDim.y;
    int array_offset = img.num_moments * t_offset;

    strided_array<float_complex> temp0(&storage[t_idx], t_offset);
    strided_array<float_complex> temp1(&storage[t_idx + array_offset], t_offset);
    strided_array<float_complex> temp2(&storage[t_idx + 2 * array_offset], t_offset);

    compute_num_moments(img, gid.x, gid.y, c_gen_params.error_threshold, temp0, temp1, temp2);

    // Make sure that the last index is set to zero
    if (gid.x == 0 && gid.y == 0)
        img.index[img.width * img.height] = 0;
}

template <typename VolumeDevice>
void generate_moments_device_helper(cut::Camera cam, const VolumeDevice &d_volume, MomentImageHost &img,
                                    const GenerationParameters &params)
{
    CHECK_CUDA(cudaMemcpyToSymbol(c_camera, &cam, sizeof(cut::Camera), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(c_gen_params, &params, sizeof(GenerationParameters), 0, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock;
    dim3 numBlocks;

    img.domain = d_volume.view.domain;
    MomentImageDevice mi(img);

    {
        auto shared_mem = mi.get_best_smem_launch_config(sizeof(float), numBlocks, threadsPerBlock);

        SCOPED_CUDA_QUERY("Raymarching");
        generate_moments_kernel<<<numBlocks, threadsPerBlock, shared_mem>>>(d_volume.view, mi.view);
    }

    if (params.compact_image)
    {
        {
            auto shared_mem = mi.get_best_smem_launch_config(sizeof(float_complex) * 3, numBlocks, threadsPerBlock);

            SCOPED_CUDA_QUERY("Determining moment subset");
            compute_num_moments_kernel<<<numBlocks, threadsPerBlock, shared_mem>>>(mi.view);
        }

        mi.compact();

        if (params.prediction_coding)
            mi.prediction_encode(params.coding_warp, params.bias);
    }

    mi.copy_back(img);
}

void generate_device(cut::Camera cam, const Particles &volume, MomentImageHost &img, const GenerationParameters &params)
{
    ParticlesDevice d_part(volume);

    generate_moments_device_helper(cam, d_part, img, params);
}

void generate_device(cut::Camera cam, const RegularGrid &volume, MomentImageHost &img,
                     const GenerationParameters &params)
{
    RegularGridDevice d_grid(volume);

    generate_moments_device_helper(cam, d_grid, img, params);
}

__global__ void reconstruct_single_scattering_cache_kernel(TransferFunctionDevice tf, MomentImageDeviceView mi,
                                                           ScatterGridDevice grid)
{
    auto gid = pixel_idx();
    if (gid.x >= mi.width || gid.y >= mi.height)
        return;

    Vec2f ndc(gid.x / static_cast<float>(mi.width) * 2.f - 1.f, gid.y / static_cast<float>(mi.height) * 2.f - 1.f);
    auto idx = mi.get_idx(gid.x, gid.y);
    auto num_moments = mi.get_num_moments(gid.x, gid.y);

    if (num_moments == 0)
        return;

    extern __shared__ float fast_storage[];
    int t_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int t_offset = blockDim.x * blockDim.y;

    strided_array<float> pmoments(&fast_storage[t_idx], t_offset);
    for (int m = 0; m < num_moments; ++m)
        pmoments[m] = mi.data[idx + m];

    if (c_rec_params.use_truncated_fourier)
    {
        TruncatedFourierReconstructionSMEM rec(num_moments, pmoments, mi.get_bounds(gid.x, gid.y));
        reconstruct_single_scattering_ray(get_eye_ray(c_camera, ndc), mi.domain, c_camera.znear, rec, tf, grid,
                                          c_rec_params);
    }
    else
    {
        MESEReconstructionSMEM rec(num_moments, pmoments, mi.get_bounds(gid.x, gid.y));
        reconstruct_single_scattering_ray(get_eye_ray(c_camera, ndc), mi.domain, c_camera.znear, rec, tf, grid,
                                          c_rec_params);
    }
}

__global__ void init_scatter_grid_kernel(ScatterGridDevice grid, float value)
{
    Vec3i gid(__umul24(blockIdx.x, blockDim.x) + threadIdx.x, __umul24(blockIdx.y, blockDim.y) + threadIdx.y,
              __umul24(blockIdx.z, blockDim.z) + threadIdx.z);

    if (gid.x >= grid.size.x || gid.y >= grid.size.y || gid.z >= grid.size.z)
        return;

    surf3Dwrite<float>(value, grid.surface, gid.x * sizeof(float), gid.y, gid.z);
}

RegularGridDevice reconstruct_single_scattering_cache_device(const SingleScatteringImageDevice &ss,
                                                             const TransferFunctionDevice &d_tf,
                                                             const ReconstructionParameters &params)
{
    CHECK_CUDA(cudaMemcpyToSymbol(c_camera, &ss.view.cam, sizeof(cut::Camera), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(c_rec_params, &params, sizeof(ReconstructionParameters), 0, cudaMemcpyHostToDevice));

    ScatterGridDevice grid(Vec3i(params.ss.cache_size_x, params.ss.cache_size_y, params.ss.cache_size_z),
                           ss.view.img.domain);

    {
        dim3 threads_per_block(8, 8, 8);
        dim3 num_blocks;
        num_blocks.x = (grid.size.x + threads_per_block.x - 1) / threads_per_block.x;
        num_blocks.y = (grid.size.y + threads_per_block.y - 1) / threads_per_block.y;
        num_blocks.z = (grid.size.z + threads_per_block.z - 1) / threads_per_block.z;
        init_scatter_grid_kernel<<<num_blocks, threads_per_block>>>(grid, 0.0f);
    }

    {
        SCOPED_CUDA_QUERY("Single-scattering cache");
        dim3 threadsPerBlock, numBlocks;
        auto shared_mem = ss.d_img.get_best_smem_launch_config(sizeof(float), numBlocks, threadsPerBlock);
        reconstruct_single_scattering_cache_kernel<<<numBlocks, threadsPerBlock, shared_mem>>>(d_tf, ss.view.img, grid);
    }

    return RegularGridDevice(grid);
}

template <typename SingleScattering>
__global__ void reconstruct_moments_kernel(TransferFunctionDevice tf, MomentImageDeviceView mi,
                                           ColorImageDeviceView cimg, SingleScattering ss)
{
    auto gid = pixel_idx();
    if (gid.x >= mi.width || gid.y >= mi.height)
        return;

    Vec2f ndc(gid.x / static_cast<float>(mi.width) * 2.f - 1.f, gid.y / static_cast<float>(mi.height) * 2.f - 1.f);

    auto idx = mi.get_idx(gid.x, gid.y);
    auto num_moments = mi.get_num_moments(gid.x, gid.y);

    if (num_moments == 0)
    {
        cimg.set_color(gid.x, gid.y, reconstruct_zero_ray(tf));
        return;
    }

    extern __shared__ float fast_storage[];
    int t_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int t_offset = blockDim.x * blockDim.y;
    int array_offset = mi.num_moments * t_offset;

    strided_array<float> pmoments(&fast_storage[t_idx], t_offset);

    for (int m = 0; m < num_moments; ++m)
        pmoments[m] = mi.data[idx + m];

    if (ss.enabled() && !ss.use_cache())
        ss.set_smem(strided_array<float>(&fast_storage[t_idx + array_offset], t_offset));

    Vec4f color;
    if (c_rec_params.use_truncated_fourier)
    {
        TruncatedFourierReconstructionSMEM rec(num_moments, pmoments, mi.get_bounds(gid.x, gid.y));
        color = reconstruct_ray(get_eye_ray(c_camera, ndc), mi.domain, c_camera.znear, rec, tf, ss, c_rec_params);
    }
    else
    {
        MESEReconstructionSMEM rec(num_moments, pmoments, mi.get_bounds(gid.x, gid.y));
        color = reconstruct_ray(get_eye_ray(c_camera, ndc), mi.domain, c_camera.znear, rec, tf, ss, c_rec_params);
    }

    cimg.set_color(gid.x, gid.y, color);
}

template <typename SingleScattering>
void reconstruct_moments_device_helper(cut::Camera cam, const MomentImageHost &mi, const TransferFunctionDevice &d_tf,
                                       ColorImageHost &cimg, SingleScattering ss,
                                       const ReconstructionParameters &params)
{
    CHECK_CUDA(cudaMemcpyToSymbol(c_camera, &cam, sizeof(cut::Camera), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(c_rec_params, &params, sizeof(ReconstructionParameters), 0, cudaMemcpyHostToDevice));

    ColorImageDevice d_cimg(cimg);

    MomentImageDevice d_mi(mi);
    d_mi.load_from(mi);

    if (!params.use_truncated_fourier)
        d_mi.prepare_moments_device(params.bias);
    else if (params.use_truncated_fourier && mi.prediction_code)
        d_mi.revert_prediction_coding();

    if (ss.enabled())
    {
        SCOPED_CUDA_QUERY("Raymarching");
        dim3 threads_per_block(8, 4);
        dim3 num_blocks;
        num_blocks.x = (mi.width + threads_per_block.x - 1) / threads_per_block.x;
        num_blocks.y = (mi.height + threads_per_block.y - 1) / threads_per_block.y;

        auto shared_mem =
            threads_per_block.x * threads_per_block.y * sizeof(float) * (mi.num_moments + ss.get_max_moments());

        assert(shared_mem < g_DeviceProp.sharedMemPerBlock);

        reconstruct_moments_kernel<<<num_blocks, threads_per_block, shared_mem>>>(d_tf, d_mi.view, d_cimg.view, ss);
    }
    else
    {
        SCOPED_CUDA_QUERY("Raymarching");
        dim3 threadsPerBlock, numBlocks;
        auto shared_mem = d_mi.get_best_smem_launch_config(sizeof(float), numBlocks, threadsPerBlock);
        reconstruct_moments_kernel<<<numBlocks, threadsPerBlock, shared_mem>>>(d_tf, d_mi.view, d_cimg.view, ss);
    }

    d_cimg.copy_back(cimg);
}

void reconstruct_device(cut::Camera cam, const MomentImageHost &mi, const TransferFunction &tf, ColorImageHost &cimg,
                        const ReconstructionParameters &params)
{
    TransferFunctionDevice d_tf(tf);
    reconstruct_moments_device_helper(cam, mi, d_tf, cimg, NoSingleScattering{}, params);
}

void reconstruct_device(cut::Camera cam, const MomentImageHost &mi, const TransferFunction &tf, ColorImageHost &cimg,
                        const SingleScatteringImageHost &ss, const ReconstructionParameters &params)
{
    TransferFunctionDevice d_tf(tf);

    SingleScatteringImageDevice d_ss(ss);
    d_ss.prepare(params.bias);
    if (params.ss.use_cache)
    {
        auto rgrid = reconstruct_single_scattering_cache_device(d_ss, d_tf, params);
        SingleScatteringCacheDevice d_cache(rgrid.view);

        reconstruct_moments_device_helper(cam, mi, d_tf, cimg, d_cache.get_view(), params);
    }
    else
        reconstruct_moments_device_helper(cam, mi, d_tf, cimg, d_ss.get_view(), params);
}

void generate_reference_reconstructed_ss_device(cut::Camera cam, const TransferFunction &tf, const RegularGrid &volume,
                                                ColorImageHost &img, const SingleScatteringImageHost &ss,
                                                const ReconstructionParameters &params)
{
    TransferFunctionDevice d_tf(tf);

    SingleScatteringImageDevice d_ss(ss);
    d_ss.prepare(params.bias);
    auto rgrid = reconstruct_single_scattering_cache_device(d_ss, d_tf, params);
    SingleScatteringCacheDevice d_cache(rgrid.view);

    RegularGridDevice d_grid(volume);
    generate_reference_device_helper(cam, d_grid, tf, img, d_cache.view, ref_from_rec_params(params));
}

void generate_reference_reconstructed_ss_device(cut::Camera cam, const TransferFunction &tf, const Particles &volume,
                                                ColorImageHost &img, const SingleScatteringImageHost &ss,
                                                const ReconstructionParameters &params)
{
    TransferFunctionDevice d_tf(tf);

    SingleScatteringImageDevice d_ss(ss);
    d_ss.prepare(params.bias);
    auto rgrid = reconstruct_single_scattering_cache_device(d_ss, d_tf, params);
    SingleScatteringCacheDevice d_cache(rgrid.view);

    ParticlesDevice d_part(volume);
    generate_reference_device_helper(cam, d_part, tf, img, d_cache.view, ref_from_rec_params(params));
}

__global__ void reconstruct_resample_kernel(MomentImageDeviceView mi, ScatterGridDevice grid)
{
    auto gid = pixel_idx();
    if (gid.x >= mi.width || gid.y >= mi.height)
        return;

    Vec2f ndc(gid.x / static_cast<float>(mi.width) * 2.f - 1.f, gid.y / static_cast<float>(mi.height) * 2.f - 1.f);
    auto idx = mi.get_idx(gid.x, gid.y);
    auto num_moments = mi.get_num_moments(gid.x, gid.y);

    if (num_moments == 0)
        return;

    extern __shared__ float fast_storage[];
    int t_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int t_offset = blockDim.x * blockDim.y;

    strided_array<float> pmoments(&fast_storage[t_idx], t_offset);
    for (int m = 0; m < num_moments; ++m)
        pmoments[m] = mi.data[idx + m];

    if (c_rec_params.use_truncated_fourier)
    {
        TruncatedFourierReconstructionSMEM rec(num_moments, pmoments, mi.get_bounds(gid.x, gid.y));
        reconstruct_resample_ray(get_eye_ray(c_camera, ndc), mi.domain, c_camera.znear, rec, grid, c_rec_params);
    }
    else
    {
        MESEReconstructionSMEM rec(num_moments, pmoments, mi.get_bounds(gid.x, gid.y));
        reconstruct_resample_ray(get_eye_ray(c_camera, ndc), mi.domain, c_camera.znear, rec, grid, c_rec_params);
    }
}

RegularGridDevice resample_volume_device(cut::Camera cam, const MomentImageHost &mi,
                                         const ResamplingReconstructionParameters &params)
{
    CHECK_CUDA(cudaMemcpyToSymbol(c_camera, &cam, sizeof(cut::Camera), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpyToSymbol(c_rec_params, &params.rec, sizeof(ReconstructionParameters), 0, cudaMemcpyHostToDevice));

    MomentImageDevice d_mi(mi);
    d_mi.load_from(mi);

    if (!params.rec.use_truncated_fourier)
        d_mi.prepare_moments_device(params.rec.bias);
    else if (params.rec.use_truncated_fourier && mi.prediction_code)
        d_mi.revert_prediction_coding();

    ScatterGridDevice grid(Vec3i(params.res_x, params.res_y, params.res_z), d_mi.view.domain);

    {
        dim3 threads_per_block(8, 8, 8);
        dim3 num_blocks;
        num_blocks.x = (grid.size.x + threads_per_block.x - 1) / threads_per_block.x;
        num_blocks.y = (grid.size.y + threads_per_block.y - 1) / threads_per_block.y;
        num_blocks.z = (grid.size.z + threads_per_block.z - 1) / threads_per_block.z;
        init_scatter_grid_kernel<<<num_blocks, threads_per_block>>>(grid, INVALID_DENSITY);
    }
    {
        SCOPED_CUDA_QUERY("Resampling");
        dim3 threadsPerBlock, numBlocks;
        auto shared_mem = d_mi.get_best_smem_launch_config(sizeof(float), numBlocks, threadsPerBlock);
        reconstruct_resample_kernel<<<numBlocks, threadsPerBlock, shared_mem>>>(d_mi.view, grid);
    }

    return RegularGridDevice(grid);
}

__global__ void reconstruct_reprojected_kernel(TransferFunctionDevice tf, MomentImageDeviceView mi,
                                               ColorImageDeviceView cimg, Camera mi_cam)
{
    auto gid = pixel_idx();
    if (gid.x >= cimg.width || gid.y >= cimg.height)
        return;

    Vec2f ndc(gid.x / static_cast<float>(cimg.width) * 2.f - 1.f, gid.y / static_cast<float>(cimg.height) * 2.f - 1.f);

    extern __shared__ float fast_storage[];
    int t_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int t_offset = blockDim.x * blockDim.y;
    strided_array<float> tmp(&fast_storage[t_idx], t_offset);

    Vec4f color = reconstruct_reprojected_ray(get_eye_ray(c_camera, ndc), mi, mi_cam, c_camera.znear, tf, tmp, c_rec_params);
    cimg.set_color(gid.x, gid.y, color);
}

void reconstruct_reprojected(cut::Camera mi_cam, const MomentImageHost &mi, const TransferFunctionDevice &d_tf,
                             ColorImageHost &cimg, const cut::Camera &view_cam, const ReconstructionParameters &params)
{
    CHECK_CUDA(cudaMemcpyToSymbol(c_camera, &view_cam, sizeof(cut::Camera), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(c_rec_params, &params, sizeof(ReconstructionParameters), 0, cudaMemcpyHostToDevice));

    ColorImageDevice d_cimg(cimg);

    MomentImageDevice d_mi(mi);
    d_mi.load_from(mi);

    if (!params.use_truncated_fourier)
        d_mi.prepare_moments_device(params.bias);
    else if (params.use_truncated_fourier && mi.prediction_code)
        d_mi.revert_prediction_coding();

    SCOPED_CUDA_QUERY("Raymarching Reprojected");
    dim3 threads_per_block(8, 4);
    dim3 num_blocks;
    num_blocks.x = (cimg.width + threads_per_block.x - 1) / threads_per_block.x;
    num_blocks.y = (cimg.height + threads_per_block.y - 1) / threads_per_block.y;
    int shared_mem = threads_per_block.x * threads_per_block.y * d_mi.view.num_moments * sizeof(float);
    reconstruct_reprojected_kernel<<<num_blocks, threads_per_block, shared_mem>>>(d_tf, d_mi.view, d_cimg.view, mi_cam);

    d_cimg.copy_back(cimg);
}

void reconstruct_resampled_device(cut::Camera cam, const MomentImageHost &mi, const TransferFunction &tf,
                                  ColorImageHost &cimg, const vector<cut::Camera> &new_cam, const string &output,
                                  Vec3f background, const ResamplingReconstructionParameters &params)
{
    if (params.use_cache)
    {
        auto rgrid = resample_volume_device(cam, mi, params);

        for (size_t i = 0; i < new_cam.size(); ++i)
        {
            generate_reference_device_helper(new_cam[i], rgrid, tf, cimg, SingleScatteringBruteForce{},
                                             ref_from_rec_params(params.rec));
            write_PPM(output + "_" + std::to_string(i) + ".ppm", cimg, background);
        }
    }
    else
    {
        for (size_t i = 0; i < new_cam.size(); ++i)
        {
            TransferFunctionDevice d_tf(tf);
            reconstruct_reprojected(cam, mi, d_tf, cimg, new_cam[i], params.rec);
            write_PPM(output + "_" + std::to_string(i) + ".ppm", cimg, background);
        }
    }
}

__global__ void reconstruct_samples_kernel(MomentImageDeviceView mi, SamplesImageDeviceView img)
{
    auto gid = pixel_idx();
    if (gid.x >= mi.width || gid.y >= mi.height)
        return;

    Vec2f ndc(gid.x / static_cast<float>(mi.width) * 2.f - 1.f, gid.y / static_cast<float>(mi.height) * 2.f - 1.f);

    auto m_idx = mi.get_idx(gid.x, gid.y);
    auto s_idx = img.get_idx(gid.x, gid.y);
    auto num_moments = mi.get_num_moments(gid.x, gid.y);

    if (num_moments == 0)
    {
        img.data[s_idx] = INVALID_SAMPLE;
        return;
    }

    extern __shared__ float fast_storage[];
    int t_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int t_offset = blockDim.x * blockDim.y;

    strided_array<float> pmoments(&fast_storage[t_idx], t_offset);

    for (int m = 0; m < num_moments; ++m)
        pmoments[m] = mi.data[m_idx + m];

    if (c_rec_params.use_truncated_fourier)
    {
        TruncatedFourierReconstructionSMEM rec(num_moments, pmoments, mi.get_bounds(gid.x, gid.y));
        reconstruct_samples_ray(get_eye_ray(c_camera, ndc), mi.domain, c_camera.znear, rec,
                                SamplesWriter(&img.data[s_idx]), c_rec_params);
    }
    else
    {
        MESEReconstructionSMEM rec(num_moments, pmoments, mi.get_bounds(gid.x, gid.y));
        reconstruct_samples_ray(get_eye_ray(c_camera, ndc), mi.domain, c_camera.znear, rec,
                                SamplesWriter(&img.data[s_idx]), c_rec_params);
    }
}

void reconstruct_samples_device(cut::Camera cam, const MomentImageHost &mi, SamplesImageHost &img,
                                const ReconstructionParameters &params)
{
    CHECK_CUDA(cudaMemcpyToSymbol(c_camera, &cam, sizeof(cut::Camera), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(c_rec_params, &params, sizeof(ReconstructionParameters), 0, cudaMemcpyHostToDevice));

    SamplesImageDevice d_img(img);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks;
    numBlocks.x = (mi.width + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (mi.height + threadsPerBlock.y - 1) / threadsPerBlock.y;

    MomentImageDevice d_mi(mi);
    d_mi.load_from(mi);

    if (!params.use_truncated_fourier)
        d_mi.prepare_moments_device(params.bias);
    else if (params.use_truncated_fourier && mi.prediction_code)
        d_mi.revert_prediction_coding();

    auto shared_mem = d_mi.get_best_smem_launch_config(sizeof(float), numBlocks, threadsPerBlock);

    reconstruct_samples_kernel<<<numBlocks, threadsPerBlock, shared_mem>>>(d_mi.view, d_img.view);

    d_img.copy_back(img);
}

template <typename Volume> __global__ void reconstruct_error_kernel(Volume volume, MomentImageDeviceView mi)
{
    auto gid = pixel_idx();
    if (gid.x >= mi.width || gid.y >= mi.height)
        return;

    Vec2f ndc(gid.x / static_cast<float>(mi.width) * 2.f - 1.f, gid.y / static_cast<float>(mi.height) * 2.f - 1.f);

    auto idx = mi.get_idx(gid.x, gid.y);
    auto num_moments = mi.get_num_moments(gid.x, gid.y);

    extern __shared__ float fast_storage[];
    int t_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int t_offset = blockDim.x * blockDim.y;

    strided_array<float> pmoments(&fast_storage[t_idx], t_offset);
    for (int m = 0; m < num_moments; ++m)
        pmoments[m] = mi.data[idx + m];

    MESEReconstructionSMEM rec(num_moments, pmoments, mi.get_bounds(gid.x, gid.y));
    auto e = reconstruct_error(volume, get_eye_ray(c_camera, ndc), c_camera.znear, rec, c_error_params);
    mi.set_error_bound(gid.x, gid.y, e);
}

template <typename VolumeDevice>
void reconstruct_error_device_helper(cut::Camera cam, const VolumeDevice &d_volume, MomentImageHost &mi,
                                     const ErrorReconstructionParameters &params)
{
    CHECK_CUDA(cudaMemcpyToSymbol(c_camera, &cam, sizeof(cut::Camera), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpyToSymbol(c_error_params, &params, sizeof(ReconstructionParameters), 0, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks;
    numBlocks.x = (mi.width + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (mi.height + threadsPerBlock.y - 1) / threadsPerBlock.y;

    MomentImageDevice d_mi(mi);
    d_mi.load_from(mi);
    d_mi.prepare_moments_device(params.bias);

    auto shared_mem = d_mi.get_best_smem_launch_config(sizeof(float), numBlocks, threadsPerBlock);

    reconstruct_error_kernel<<<numBlocks, threadsPerBlock, shared_mem>>>(d_volume.view, d_mi.view);

    d_mi.copy_back_errors(mi);
}

void reconstruct_error_device(cut::Camera cam, const Particles &volume, MomentImageHost &mi,
                              const ErrorReconstructionParameters &params)
{
    ParticlesDevice d_part(volume);
    reconstruct_error_device_helper(cam, d_part, mi, params);
}

void reconstruct_error_device(cut::Camera cam, const RegularGrid &volume, MomentImageHost &mi,
                              const ErrorReconstructionParameters &params)
{
    RegularGridDevice d_grid(volume);
    reconstruct_error_device_helper(cam, d_grid, mi, params);
}

template <typename SingleScattering>
__global__ void reconstruct_uncertainty_interpolation_kernel(TransferFunctionDevice tf, MomentImageDeviceView mi,
                                                             ColorImageDeviceView cimg, SingleScattering ss,
                                                             float bound_interpolation)
{
    auto gid = pixel_idx();
    if (gid.x >= mi.width || gid.y >= mi.height)
        return;

    Vec2f ndc(gid.x / static_cast<float>(mi.width) * 2.f - 1.f, gid.y / static_cast<float>(mi.height) * 2.f - 1.f);

    auto idx = mi.get_idx(gid.x, gid.y);
    auto num_moments = mi.get_num_moments(gid.x, gid.y);

    if (num_moments == 0)
    {
        cimg.set_color(gid.x, gid.y, reconstruct_zero_ray(tf));
        return;
    }

    Vec4f color;

    extern __shared__ float fast_storage[];
    int t_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int t_offset = blockDim.x * blockDim.y;

    strided_array<float> pmoments(&fast_storage[t_idx], t_offset);
    for (int m = 0; m < num_moments; ++m)
        pmoments[m] = mi.data[idx + m];

    if (c_rec_params.use_truncated_fourier)
    {
        TruncatedFourierReconstructionSMEM mese(num_moments, pmoments, mi.get_bounds(gid.x, gid.y));
        ErrorBoundMESEReconstruction<TruncatedFourierReconstructionSMEM> bound_rec(mese,
                                                                                   mi.get_error_bound(gid.x, gid.y));
        ErrorBoundInterpolator<ErrorBoundMESEReconstruction<TruncatedFourierReconstructionSMEM>> bound_interp(
            bound_rec, bound_interpolation);
        color =
            reconstruct_ray(get_eye_ray(c_camera, ndc), mi.domain, c_camera.znear, bound_interp, tf, ss, c_rec_params);
    }
    else
    {
        MESEReconstructionSMEM mese(num_moments, pmoments, mi.get_bounds(gid.x, gid.y));
        ErrorBoundMESEReconstruction<MESEReconstructionSMEM> bound_rec(mese, mi.get_error_bound(gid.x, gid.y));
        ErrorBoundInterpolator<ErrorBoundMESEReconstruction<MESEReconstructionSMEM>> bound_interp(bound_rec,
                                                                                                  bound_interpolation);
        color =
            reconstruct_ray(get_eye_ray(c_camera, ndc), mi.domain, c_camera.znear, bound_interp, tf, ss, c_rec_params);
    }

    cimg.set_color(gid.x, gid.y, color);
}

template <typename SingleScattering>
void reconstruct_uncertainty_interpolation_device_helper(cut::Camera cam, const MomentImageHost &mi,
                                                         const TransferFunction &tf, ColorImageHost &cimg,
                                                         SingleScattering ss,
                                                         const UncertaintyReconstructionParameters &params)
{
    CHECK_CUDA(cudaMemcpyToSymbol(c_camera, &cam, sizeof(cut::Camera), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpyToSymbol(c_rec_params, &params.rec, sizeof(ReconstructionParameters), 0, cudaMemcpyHostToDevice));

    ColorImageDevice d_cimg(cimg);
    TransferFunctionDevice d_tf(tf);

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks;
    numBlocks.x = (mi.width + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (mi.height + threadsPerBlock.y - 1) / threadsPerBlock.y;

    MomentImageDevice d_mi(mi);
    d_mi.load_from(mi);

    if (!params.rec.use_truncated_fourier)
        d_mi.prepare_moments_device(params.rec.bias);
    else if (params.rec.use_truncated_fourier && mi.prediction_code)
        d_mi.revert_prediction_coding();

    auto shared_mem = d_mi.get_best_smem_launch_config(sizeof(float), numBlocks, threadsPerBlock);
    reconstruct_uncertainty_interpolation_kernel<<<numBlocks, threadsPerBlock, shared_mem>>>(
        d_tf, d_mi.view, d_cimg.view, ss, params.bound_interpolation);

    d_cimg.copy_back(cimg);
}

void reconstruct_uncertainty_interpolation_device(cut::Camera cam, const MomentImageHost &mi,
                                                  const TransferFunction &tf, ColorImageHost &cimg,
                                                  const UncertaintyReconstructionParameters &params)
{
    reconstruct_uncertainty_interpolation_device_helper(cam, mi, tf, cimg, NoSingleScattering{}, params);
}

void reconstruct_uncertainty_interpolation_device(cut::Camera cam, const MomentImageHost &mi,
                                                  const TransferFunction &tf, ColorImageHost &cimg,
                                                  const SingleScatteringImageHost &ss,
                                                  const UncertaintyReconstructionParameters &params)
{
    SingleScatteringImageDevice d_ss(ss);
    d_ss.prepare(params.rec.bias);
    reconstruct_uncertainty_interpolation_device_helper(cam, mi, tf, cimg, d_ss.get_view(), params);
}

template <typename SingleScattering>
__global__ void reconstruct_uncertainty_convolution_kernel(TransferFunctionDevice tf, MomentImageDeviceView mi,
                                                           ColorImageDeviceView cimg, SingleScattering ss,
                                                           int error_type)
{
    auto gid = pixel_idx();
    if (gid.x >= mi.width || gid.y >= mi.height)
        return;

    Vec2f ndc(gid.x / static_cast<float>(mi.width) * 2.f - 1.f, gid.y / static_cast<float>(mi.height) * 2.f - 1.f);

    auto idx = mi.get_idx(gid.x, gid.y);
    auto num_moments = mi.get_num_moments(gid.x, gid.y);

    if (num_moments == 0)
    {
        cimg.set_color(gid.x, gid.y, reconstruct_zero_ray(tf));
        return;
    }

    extern __shared__ float fast_storage[];
    int t_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int t_offset = blockDim.x * blockDim.y;

    strided_array<float> pmoments(&fast_storage[t_idx], t_offset);
    for (int m = 0; m < num_moments; ++m)
        pmoments[m] = mi.data[idx + m];

    Vec4f color;
    if (error_type == MAX_ERROR && c_rec_params.use_truncated_fourier)
    {
        TruncatedFourierReconstructionSMEM mese(num_moments, pmoments, mi.get_bounds(gid.x, gid.y));
        ErrorBoundMESEReconstruction<TruncatedFourierReconstructionSMEM> bound_rec(mese,
                                                                                   mi.get_error_bound(gid.x, gid.y));
        color = reconstruct_ray_uncertainty_tf(get_eye_ray(c_camera, ndc), mi.domain, c_camera.znear, bound_rec, tf, ss,
                                               c_rec_params);
    }
    else if (error_type == RMSE && c_rec_params.use_truncated_fourier)
    {
        TruncatedFourierReconstructionSMEM mese(num_moments, pmoments, mi.get_bounds(gid.x, gid.y));
        MeanStdDevMESEReconstruction<TruncatedFourierReconstructionSMEM> bound_rec(mese,
                                                                                   mi.get_error_bound(gid.x, gid.y));
        color = reconstruct_ray_uncertainty_tf(get_eye_ray(c_camera, ndc), mi.domain, c_camera.znear, bound_rec, tf, ss,
                                               c_rec_params);
    }
    else if (error_type == MAX_ERROR && !c_rec_params.use_truncated_fourier)
    {
        MESEReconstructionSMEM mese(num_moments, pmoments, mi.get_bounds(gid.x, gid.y));
        ErrorBoundMESEReconstruction<MESEReconstructionSMEM> bound_rec(mese, mi.get_error_bound(gid.x, gid.y));
        color = reconstruct_ray_uncertainty_tf(get_eye_ray(c_camera, ndc), mi.domain, c_camera.znear, bound_rec, tf, ss,
                                               c_rec_params);
    }
    else
    {
        MESEReconstructionSMEM mese(num_moments, pmoments, mi.get_bounds(gid.x, gid.y));
        MeanStdDevMESEReconstruction<MESEReconstructionSMEM> bound_rec(mese, mi.get_error_bound(gid.x, gid.y));
        color = reconstruct_ray_uncertainty_tf(get_eye_ray(c_camera, ndc), mi.domain, c_camera.znear, bound_rec, tf, ss,
                                               c_rec_params);
    }

    cimg.set_color(gid.x, gid.y, color);
}

void reconstruct_uncertainty_convolution_device(cut::Camera cam, const MomentImageHost &mi, const TransferFunction &tf,
                                                ColorImageHost &cimg, const UncertaintyReconstructionParameters &params)
{
    CHECK_CUDA(cudaMemcpyToSymbol(c_camera, &cam, sizeof(cut::Camera), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpyToSymbol(c_rec_params, &params.rec, sizeof(ReconstructionParameters), 0, cudaMemcpyHostToDevice));

    ColorImageDevice d_cimg(cimg);
    TransferFunctionDevice d_tf(tf);

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks;
    numBlocks.x = (mi.width + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (mi.height + threadsPerBlock.y - 1) / threadsPerBlock.y;

    MomentImageDevice d_mi(mi);
    d_mi.load_from(mi);

    if (!params.rec.use_truncated_fourier)
        d_mi.prepare_moments_device(params.rec.bias);
    else if (params.rec.use_truncated_fourier && mi.prediction_code)
        d_mi.revert_prediction_coding();

    auto shared_mem = d_mi.get_best_smem_launch_config(sizeof(float), numBlocks, threadsPerBlock);

    reconstruct_uncertainty_convolution_kernel<<<numBlocks, threadsPerBlock, shared_mem>>>(
        d_tf, d_mi.view, d_cimg.view, NoSingleScattering{}, params.error_type);

    d_cimg.copy_back(cimg);
}

template <typename RandomSampler>
__global__ void reconstruct_rayhistogram_kernel(TransferFunctionDevice tf, RayHistogramImageDeviceView img,
                                                RandomSampler sampler, ColorImageDeviceView cimg)
{
    auto gid = pixel_idx();
    if (gid.x >= img.width || gid.y >= img.height)
        return;

    Vec2f ndc(gid.x / static_cast<float>(img.width) * 2.f - 1.f, gid.y / static_cast<float>(img.height) * 2.f - 1.f);

    sampler.setup(gid.x, gid.y);

    Vec4f color;
    color = reconstruct_rayhistogram_ray(get_eye_ray(c_camera, ndc), img.domain, c_camera.znear,
                                         img.get_view(gid.x, gid.y), sampler, tf, c_rayhist_rec_params);

    cimg.set_color(gid.x, gid.y, color);
}

void reconstruct_rayhistogram_device(cut::Camera cam, const RayHistogramImageHost &img, const TransferFunction &tf,
                                     ColorImageHost &cimg, const RayHistogramReconstructionParameters &params)
{
    CHECK_CUDA(cudaMemcpyToSymbol(c_camera, &cam, sizeof(cut::Camera), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(c_rayhist_rec_params, &params, sizeof(RayHistogramReconstructionParameters), 0,
                                  cudaMemcpyHostToDevice));

    ColorImageDevice d_cimg(cimg);
    TransferFunctionDevice d_tf(tf);

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks;
    numBlocks.x = (img.width + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (img.height + threadsPerBlock.y - 1) / threadsPerBlock.y;

    RayHistogramImageDevice d_img(img);

    SimpleRayHistogramSamplerDevice d_sampler;
    d_sampler.init(img.width, img.height);

    {
        SCOPED_CUDA_QUERY("Sample ray-histograms");
        reconstruct_rayhistogram_kernel<<<numBlocks, threadsPerBlock>>>(d_tf, d_img.view, d_sampler, d_cimg.view);
    }

    d_sampler.destroy();
    d_cimg.copy_back(cimg);
}

__global__ void reconstruct_rayhistogram_samples_kernel(RayHistogramImageDeviceView img,
                                                        SimpleRayHistogramSamplerDevice sampler,
                                                        SamplesImageDeviceView simg)
{
    auto gid = pixel_idx();
    if (gid.x >= img.width || gid.y >= img.height)
        return;

    Vec2f ndc(gid.x / static_cast<float>(img.width) * 2.f - 1.f, gid.y / static_cast<float>(img.height) * 2.f - 1.f);

    auto idx = simg.get_idx(gid.x, gid.y);

    sampler.setup(gid.x, gid.y);
    reconstruct_rayhistogram_samples_ray(get_eye_ray(c_camera, ndc), img.domain, c_camera.znear,
                                         img.get_view(gid.x, gid.y), sampler, SamplesWriter(&simg.data[idx]),
                                         c_rayhist_rec_params);
}

void reconstruct_rayhistogram_samples_device(cut::Camera cam, const RayHistogramImageHost &img, SamplesImageHost &simg,
                                             const RayHistogramReconstructionParameters &params)
{
    CHECK_CUDA(cudaMemcpyToSymbol(c_camera, &cam, sizeof(cut::Camera), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(c_rayhist_rec_params, &params, sizeof(RayHistogramReconstructionParameters), 0,
                                  cudaMemcpyHostToDevice));

    SamplesImageDevice d_simg(simg);

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks;
    numBlocks.x = (img.width + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (img.height + threadsPerBlock.y - 1) / threadsPerBlock.y;

    RayHistogramImageDevice d_img(img);

    SimpleRayHistogramSamplerDevice d_sampler;
    d_sampler.init(img.width, img.height);

    reconstruct_rayhistogram_samples_kernel<<<numBlocks, threadsPerBlock>>>(d_img.view, d_sampler, d_simg.view);

    d_sampler.destroy();
    d_simg.copy_back(simg);
}