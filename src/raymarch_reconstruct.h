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
#ifndef MRAY_RAYMARCH_RECONSTRUCT_H
#define MRAY_RAYMARCH_RECONSTRUCT_H

#include "raymarch_common.h"
#include "cut/phase_functions.h"

#include "reconstruction_techniques.h"
#include "error_measures.h"

/**
 * Computes transmittance by sampling cached transmittance.
 */
template <typename TransferFunction, typename SingleScattering>
FUNC std::enable_if_t<is_single_scattering_cached<SingleScattering>::value, float>
compute_transmittance(const SingleScattering &ss, const TransferFunction &tf, Range<Vec3f> domain, Vec3f pt,
                      const ReconstructionParameters &rec_params)
{
    UNUSED(tf);
    UNUSED(domain);
    UNUSED(rec_params);
    return ss.sample_cache(pt);
}

/**
 * Computes transmittance by ray marching a (single scattering) moment image.
 *
 * FIXME: We gave up on this code and use the SingleScatteringCache instead (resampling to a grid).
 * It is unclear if this code works correctly!!
 */
template <typename TransferFunction, typename SingleScattering>
FUNC std::enable_if_t<!is_single_scattering_cached<SingleScattering>::value, float>
compute_transmittance(const SingleScattering &ss, const TransferFunction &tf, Range<Vec3f> domain, Vec3f pt,
                      const ReconstructionParameters &rec_params)
{
    auto ndc = ss.get_ndc(pt);

    // The ray is FROM the light source
    auto ray = ss.get_ray(ndc);

    float t_near, t_far;
    auto hit = intersect_AABB(ray, domain.min, domain.max, t_near, t_far);
    if (!hit || t_far <= 0.0f || t_near >= t_far)
        return 0.0f;

    float ss_step_size = rec_params.step_size * rec_params.ss.step_size_factor;
    float phase_step = get_phase_step(t_near, t_far, ss_step_size);

    auto t = cut::min(t_far, cut::length(pt - ray.origin));
    float p = to_phase(t_near, t_far, t);

    auto rec = ss.get_reconstruction(ndc);

    float fs = rec.reconstruct(p);
    float bs;

    float tau = 0.0f;
    while (t >= t_near)
    {
        // Ray march in reverse direction TO the light source
        t -= ss_step_size;
        p -= phase_step;

        bs = rec.reconstruct(p);
        float sigma_t = tf.sample(fs, bs).w;

        fs = bs;
        tau += sigma_t;
    }

    return exp(-tau * rec_params.ss.step_size_factor);
}

template <typename TransferFunction, typename SingleScattering>
FUNC std::enable_if_t<std::is_same<SingleScattering, NoSingleScattering>::value, Vec3f>
inscattering(const SingleScattering &ss, const TransferFunction &tf, Range<Vec3f> domain, cut::Ray ray, float t,
             const ReconstructionParameters &rec_params)
{
    UNUSED(ss);
    UNUSED(tf);
    UNUSED(domain);
    UNUSED(ray);
    UNUSED(t);
    UNUSED(rec_params);
    return Vec3f(0.f);
}

/**
 * Computes in scattering due to single scattering.
 * Applies a Henyey-Greenstein phase function.
 */
template <typename TransferFunction, typename SingleScattering>
FUNC std::enable_if_t<!std::is_same<SingleScattering, NoSingleScattering>::value, Vec3f>
inscattering(const SingleScattering &ss, const TransferFunction &tf, Range<Vec3f> domain, cut::Ray ray, float t,
             const ReconstructionParameters &rec_params)
{
    float T_ss = compute_transmittance(ss, tf, domain, ray.origin + t * ray.dir, rec_params);

    Vec3f light_dir(rec_params.ss.direction_x, rec_params.ss.direction_y, rec_params.ss.direction_z);
    float pf = henyey_greenstein(dot(-ray.dir, light_dir), rec_params.ss.henyey_greenstein_g);
    return T_ss * pf * Vec3f(rec_params.ss.intensity_r, rec_params.ss.intensity_g, rec_params.ss.intensity_b);
}

template <typename TransferFunction> FUNC Vec4f reconstruct_zero_ray(const TransferFunction &tf)
{
    Vec4f rgba = tf.sample(0.0f, 0.0f);
    return Vec4f(rgba.x, rgba.y, rgba.z, 0.0f);
}

/**
 * Reconstructs the signal along a ray with a given reconstruction method
 * @tparam ReconstructionTechnique: Can be the bounded MESE, Fourier, ... (see reconstruction_technique.h)
 */
template <typename ReconstructionTechnique, typename TransferFunction, typename SingleScattering>
FUNC Vec4f reconstruct_ray(cut::Ray ray, Range<Vec3f> domain, float znear, ReconstructionTechnique rec,
                           const TransferFunction &tf, const SingleScattering &ss,
                           const ReconstructionParameters &rec_params)
{
    float t_near, t_far;
    auto hit = intersect_AABB(ray, domain.min, domain.max, t_near, t_far);
    t_near = cut::max(t_near, znear);

    auto trange = cut::Vec2f(t_near, t_far);

    float world_step_size = rec_params.step_size;

    if (hit && (t_far > 0))
    {
        float t = t_near;
        float p = to_phase(trange.x, trange.y, t);
        float phase_step = get_phase_step(trange.x, trange.y, world_step_size);

        Vec3f L(0.0f);
        float alpha = 0.0f;
        float T = 1.0f;
        float tau = 0.0f;

        float fs = rec.reconstruct(p);
        float bs;

        while (t <= t_far)
        {
            t += world_step_size;
            p += phase_step;

            bs = rec.reconstruct(p);

            Vec4f rgba = tf.sample(fs, bs);
            float sigma_t = rgba.w;

            Vec3f Le = rec_params.medium_emission * Vec3f(rgba.x, rgba.y, rgba.z);

            Vec3f Ls = Vec3f(0.0f);
            float sigma_s = rec_params.medium_scattering;
            if (ss.enabled() && sigma_s > 0.0f)
                Ls = inscattering(ss, tf, domain, ray, t, rec_params);

            // compositing with preintegration(!)
            L += (Le + Ls * sigma_s) * T;
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
 * Reconstructs the signal along a ray and incorporates the uncertainty bounds for transfer function classification.
 * @tparam ReconstructionTechnique: Can be the bounded MESE, Fourier, ... (see reconstruction_technique.h)
 */
template <typename ReconstructionTechnique, typename TransferFunction, typename SingleScattering>
FUNC Vec4f reconstruct_ray_uncertainty_tf(cut::Ray ray, Range<Vec3f> domain, float znear, ReconstructionTechnique rec,
                                          const TransferFunction &tf, const SingleScattering &ss,
                                          const ReconstructionParameters &rec_params)
{
    float t_near, t_far;
    auto hit = intersect_AABB(ray, domain.min, domain.max, t_near, t_far);
    t_near = cut::max(t_near, znear);

    auto trange = cut::Vec2f(t_near, t_far);

    float world_step_size = rec_params.step_size;

    if (hit && (t_far > 0))
    {
        float t = t_near;
        float p = to_phase(trange.x, trange.y, t);
        float phase_step = get_phase_step(trange.x, trange.y, world_step_size);

        Vec3f L(0.0f);
        float alpha = 0.0f;
        float T = 1.0f;
        float tau = 0.0f;

        while (t <= t_far)
        {
            t += world_step_size;
            p += phase_step;

            auto density = rec.reconstruct_bounds(p);

            Vec4f rgba = tf.sample(density.x, density.y);
            float sigma_t = rgba.w;

            Vec3f Le = rec_params.medium_emission * Vec3f(rgba.x, rgba.y, rgba.z);

            Vec3f Ls = Vec3f(0.0f);
            float sigma_s = rec_params.medium_scattering;
            if (ss.enabled() && sigma_s > 0.0f)
                Ls = inscattering(ss, tf, domain, ray, t, rec_params);

            // compositing with Preintegration(!)
            L += (Le + Ls * sigma_s) * T;
            tau += sigma_t;
            T = exp(-tau);
            alpha = 1.0f - T;

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
 * Reconstructs the signal along a ray and writes samples to memory (for debugging/evaluation).
 * @tparam ReconstructionTechnique: Can be the bounded MESE, Fourier, ... (see reconstruction_technique.h)
 */
template <typename ReconstructionTechnique>
FUNC void reconstruct_samples_ray(cut::Ray ray, Range<Vec3f> domain, float znear, ReconstructionTechnique rec,
                                  SamplesWriter samples, const ReconstructionParameters &rec_params)
{
    float t_near, t_far;
    auto hit = intersect_AABB(ray, domain.min, domain.max, t_near, t_far);
    t_near = cut::max(t_near, znear);

    auto trange = cut::Vec2f(t_near, t_far);

    float world_step_size = rec_params.step_size_write_samples;

    int i = 0;
    if (hit && (t_far > 0))
    {
        float t = t_near;
        float p = to_phase(trange.x, trange.y, t);
        float phase_step = get_phase_step(trange.x, trange.y, world_step_size);

        float fs = rec.reconstruct(p);
        float bs;

        while (t < t_far && i < samples.get_max())
        {
            t += world_step_size;
            p += phase_step;

            bs = rec.reconstruct(p);

            samples.write(i, fs);
            fs = bs;
            ++i;
        }

        // Add last sample exactly at t_far
        if (i < samples.get_max())
        {
            samples.write(i, rec.reconstruct(get_max_phase()));
            ++i;
        }
    }

    if (i < samples.get_max())
    {
        samples.write_invalid(i);
        ++i;
    }
}

/**
 * Reconstructs the signal and samples the volume along a ray to measure an error.
 */
template <typename ReconstructionTechnique, typename Volume, typename ErrorMeasure, typename Parameters>
FUNC float reconstruct_error(const Volume &volume, cut::Ray ray, float znear, ReconstructionTechnique rec,
                             const Parameters &params, ErrorMeasure error)
{
    float t_near, t_far;
    auto hit = intersect_AABB(ray, volume.domain.min, volume.domain.max, t_near, t_far);
    t_near = cut::max(t_near, znear);

    if (hit && (t_far > 0))
    {
        auto trange = cut::Vec2f(t_near, t_far);
        float world_step_size = params.step_size;

        float t = t_near;
        while (t <= t_far)
        {
            float p = to_phase(trange.x, trange.y, t);

            auto fs = rec.reconstruct(p);
            auto ref = map_density(volume.sample(ray.origin + t * ray.dir), params);

            error.update(ref, fs);

            t += world_step_size;
        }

        return error.error();
    }
    else
        return 0.0f;
}

/**
 * Reconstructs the signal and samples the volume along a ray to measure an error.
 */
template <typename ReconstructionTechnique, typename Volume, typename Parameters>
FUNC float reconstruct_error(const Volume &volume, cut::Ray ray, float znear, ReconstructionTechnique rec,
                             const Parameters &params)
{
    if (params.error_type == MAX_ERROR)
    {
        return reconstruct_error(volume, ray, znear, rec, params, ErrorMax{});
    }
    else if (params.error_type == RMSE)
    {
        return reconstruct_error(volume, ray, znear, rec, params, ErrorRMSE{});
    }
    else if (params.error_type == ERROR_PERCENTILE)
    {
        return reconstruct_error(volume, ray, znear, rec, params, ErrorPercentile(params.error_percentile));
    }
    else
    {
        assert(false);
        return 0.0f;
    }
}

#endif // MRAY_RAYMARCH_RECONSTRUCT_H
