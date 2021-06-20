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
#ifndef MRAY_RAYMARCH_RESAMPLE_H
#define MRAY_RAYMARCH_RESAMPLE_H

#include "raymarch_common.h"
#include "reconstruction_techniques.h"

FUNC cut::Vec2i get_nearest_pixel(Vec2f ndc, int w, int h)
{
    return cut::Vec2i(cut::clamp(static_cast<int>((ndc.x * 0.5f + 0.5f) * w), 0, w - 1),
                      cut::clamp(static_cast<int>((ndc.y * 0.5f + 0.5f) * h), 0, h - 1));
}

/** Reconstruct the scalar density at a single point in space by reprojecting a moment image */
template <typename MomentImage, typename float_array>
FUNC float reconstruct_reprojected_density(MomentImage &mi, const cut::Camera &mi_cam, Vec3f p, float_array tmp_mem,
                                           bool use_fourier)
{
    cut::Ray ray;
    ray.origin = mi_cam.position;
    ray.dir = cut::normalize(p - ray.origin);

    float t_near, t_far;
    auto hit = intersect_AABB(ray, mi.domain.min, mi.domain.max, t_near, t_far);
    if (!hit || t_far <= 0.0f || t_near >= t_far)
        return 0.0f;

    t_near = cut::max(t_near, mi_cam.znear);

    auto t = cut::length(p - ray.origin);
    float phase = to_phase(t_near, t_far, t);

    static constexpr float eps = 1e-5f;
    if (phase < (-M_PI - eps) || phase > eps)
        return INVALID_DENSITY;

    phase = cut::clamp(phase, -static_cast<float>(M_PI), 0.0f);

    auto proj = mi_cam.inv_view_transform * Vec4f(ray.dir, 0.0f);
    Vec2f ndc(- proj.x / (proj.z * mi_cam.aspect * mi_cam.scale), - proj.y / (proj.z * mi_cam.scale));

    if (ndc.x < - 1.0f || ndc.x > 1.0f || ndc.y < - 1.0f || ndc.y > 1.0f)
        return INVALID_DENSITY;

//#define RESAMPLING_NEAREST_NEIGHBOR
#ifdef RESAMPLING_NEAREST_NEIGHBOR
    auto pi = get_nearest_pixel(ndc, mi.width, mi.height);
    auto num_moments = mi.get_num_moments(pi.x, pi.y);
    if (num_moments == 0)
        return 0.0f;

    auto mi_idx = mi.get_idx(pi.x, pi.y);
    auto bounds = mi.get_bounds(pi.x, pi.y);

    for (int m = 0; m < num_moments; ++m)
        tmp_mem[m] = mi.data[mi_idx + m];
#else
    // Bilinear interpolation
    Vec2f ip((ndc.x * 0.5f + 0.5f) * mi.width, (ndc.y * 0.5f + 0.5f) * mi.height);
    Vec2f a = ip - Vec2f(std::floor(ip.x), std::floor(ip.y));

    auto i_x = cut::clamp(static_cast<int>(std::floor(ip.x)), 0, mi.width - 1);
    auto i_y = cut::clamp(static_cast<int>(std::floor(ip.y)), 0, mi.height - 1);
    auto i_x_1 = cut::clamp(static_cast<int>(i_x) + 1, 0, mi.width - 1);
    auto i_y_1 = cut::clamp(static_cast<int>(i_y) + 1, 0, mi.height - 1);

    auto num_moments_x = cut::max(mi.get_num_moments(i_x, i_y), mi.get_num_moments(i_x_1, i_y));
    auto num_moments_y = cut::max(mi.get_num_moments(i_x, i_y_1), mi.get_num_moments(i_x_1, i_y_1));
    auto num_moments = cut::max(num_moments_x, num_moments_y);

    auto idx = mi.get_idx(i_x, i_y);
    for (int m = 0; m < mi.get_num_moments(i_x, i_y); ++m)
        tmp_mem[m] = (1.f - a.x) * (1.f - a.y) * mi.data[idx + m];

    for (int m = mi.get_num_moments(i_x, i_y); m < num_moments; ++m)
        tmp_mem[m] = 0;

    idx = mi.get_idx(i_x_1, i_y);
    for (int m = 0; m < mi.get_num_moments(i_x_1, i_y); ++m)
        tmp_mem[m] += a.x * (1.f - a.y) * mi.data[idx + m];

    idx = mi.get_idx(i_x, i_y_1);
    for (int m = 0; m < mi.get_num_moments(i_x, i_y_1); ++m)
        tmp_mem[m] += (1.f - a.x) * a.y * mi.data[idx + m];

    idx = mi.get_idx(i_x_1, i_y_1);
    for (int m = 0; m < mi.get_num_moments(i_x_1, i_y_1); ++m)
        tmp_mem[m] += a.x * a.y * mi.data[idx + m];

    Vec2f bounds = (1.f - a.x) * (1.f - a.y) * mi.get_bounds(i_x, i_y) + a.x * (1.f - a.y) * mi.get_bounds(i_x_1, i_y) +
                   (1.f - a.x) * a.y * mi.get_bounds(i_x, i_y_1) + a.x * a.y * mi.get_bounds(i_x_1, i_y_1);
#endif

#ifdef __CUDA_ARCH__
    if (use_fourier)
        return TruncatedFourierReconstructionSMEM(num_moments, tmp_mem, bounds).reconstruct(phase);
    else
        return MESEReconstructionSMEM(num_moments, tmp_mem, bounds).reconstruct(phase);
#else
    if (use_fourier)
        return TruncatedFourierReconstructionDefault(num_moments, tmp_mem, bounds).reconstruct(phase);
    else
        return MESEReconstructionDefault(num_moments, tmp_mem, bounds).reconstruct(phase);
#endif
}

/**
 * Reconstructs the signal along a ray from a changed camera. Uses re-projection to the given moment image.
 * @tparam ReconstructionTechnique: Can be the bounded MESE, Fourier, ... (see reconstruction_technique.h)
 */
template <typename MomentImage, typename TransferFunction, typename float_array>
FUNC Vec4f reconstruct_reprojected_ray(cut::Ray ray, MomentImage &mi, const cut::Camera &mi_cam, float znear,
                                       const TransferFunction &tf, float_array tmp_mem,
                                       const ReconstructionParameters &rec_params)
{
    float t_near, t_far;
    auto hit = intersect_AABB(ray, mi.domain.min, mi.domain.max, t_near, t_far);
    t_near = cut::max(t_near, znear);

    float world_step_size = rec_params.step_size;

    if (hit && (t_far > 0) && (t_near < t_far))
    {
        float t = t_near;

        Vec3f L(0.0f);
        float alpha = 0.0f;
        float T = 1.0f;
        float tau = 0.0f;

        Vec3f p = ray.origin + t * ray.dir;

        float fs = reconstruct_reprojected_density(mi, mi_cam, p, tmp_mem, rec_params.use_truncated_fourier);
        float bs;

        while (t <= t_far)
        {
            t += world_step_size;
            p = ray.origin + t * ray.dir;

            bs = reconstruct_reprojected_density(mi, mi_cam, p, tmp_mem, rec_params.use_truncated_fourier);

            Vec4f rgba = tf.sample(fs, bs);
            if (fs >= 1e10f && bs >= 1e10f) // INVALID
                rgba = Vec4f(0, 1, 0, 0.01f);

            float sigma_t = rgba.w;

            Vec3f Le = rec_params.medium_emission * Vec3f(rgba.x, rgba.y, rgba.z);

            // compositing with Preintegration(!)
            L += Le * T;
            tau += sigma_t;
            T = exp(-tau);
            alpha = 1.0f - T;

            fs = bs;

            if (alpha >= rec_params.early_ray_termination)
            {
                alpha = 1.f;
                break;
            }
        }

        return Vec4f(L, alpha);
    }
    else
        return Vec4f(0.f);
}

/**
 * Ray marching to reconstruct a signal using a moment image, then scatter the density to a grid.
 * TODO: Some form of DDA or grid traversal would be better.
 */
template <typename ReconstructionTechnique, typename ScatterGrid>
FUNC void reconstruct_resample_ray(cut::Ray ray, Range<Vec3f> domain, float znear, ReconstructionTechnique rec,
                                   ScatterGrid &grid, const ReconstructionParameters &rec_params)
{
    float t_near, t_far;
    auto hit = intersect_AABB(ray, domain.min, domain.max, t_near, t_far);
    t_near = cut::max(t_near, znear);

    auto trange = cut::Vec2f(t_near, t_far);

    float world_step_size = rec_params.step_size * 0.1f;

    if (hit && (t_far > 0))
    {
        float t = t_near;
        float p = to_phase(trange.x, trange.y, t);
        float phase_step = get_phase_step(trange.x, trange.y, world_step_size);

        float s = 0.0f;
        float w = 0.0f;
        Vec3i v = grid.get_voxel(ray.origin + ray.dir * t_near);

        while (t <= t_far)
        {
            t += world_step_size;
            p += phase_step;

            s += rec.reconstruct(p) * (rec_params.data_max - rec_params.data_min) + rec_params.data_min;
            w += 1.f;

            Vec3i next_v = grid.get_voxel(ray.origin + ray.dir * t);
            if (v != next_v)
            {
                v = next_v;
                grid.scatter(ray.origin + t * ray.dir, s / w);
                s = 0.0f;
                w = 0.0f;
            }
        }
    }
}

#endif // MRAY_RAYMARCH_RESAMPLE_H
