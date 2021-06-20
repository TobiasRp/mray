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
#ifndef MRAY_RAYMARCH_REFERENCE_H
#define MRAY_RAYMARCH_REFERENCE_H

#include "cut/raytracing.h"
#include "cut/phase_functions.h"
#include "parameters.h"

struct SingleScatteringBruteForce
{
};

/** "Brute-force" single scattering by ray marching secondary rays
 * @return Computed transmittance.
 */
template <typename Volume, typename TransferFunc, typename SingleScattering>
FUNC std::enable_if_t<std::is_same<SingleScattering, SingleScatteringBruteForce>::value, float>
compute_transmittance(const Volume &volume, const TransferFunc &tf, const SingleScattering &ss, cut::Ray ray,
                      const ReferenceParameters &ref_params)
{
    UNUSED(ss);
    float t_near, t_far;
    auto hit = intersect_AABB(ray, volume.domain.min, volume.domain.max, t_near, t_far);
    if (!hit || t_far <= 0.0f || t_near >= t_far)
        return 0.0f;

    t_near = cut::max(t_near, 0.0f);

    float t = t_near;
    float fs = map_density(volume.sample(ray.origin + t * ray.dir), ref_params);
    float bs;

    float ss_step_size = ref_params.step_size * ref_params.ss.step_size_factor;

    float tau = 0.0f;
    while (t <= t_far)
    {
        t += ss_step_size;

        bs = map_density(volume.sample(ray.origin + t * ray.dir), ref_params);

        float sigma_t = tf.sample(fs, bs).w;

        fs = bs;
        tau += sigma_t;
    }

    return exp(-tau * ref_params.ss.step_size_factor);
}

/** Single scattering by sampling cached transmittance.
 * @return Cached transmittance.
 */
template <typename Volume, typename TransferFunc, typename SingleScattering>
FUNC std::enable_if_t<is_single_scattering_cached<SingleScattering>::value, float>
compute_transmittance(const Volume &volume, const TransferFunc &tf, const SingleScattering &ss, cut::Ray ray,
                      const ReferenceParameters &ref_params)
{
    UNUSED(tf);
    UNUSED(volume);
    UNUSED(ref_params);
    return ss.sample_cache(ray.origin);
}

/**
 * Reference direct volume rendering implementation.
 * @return Composited color.
 */
template <typename Volume, typename TransferFunc, typename SingleScattering>
FUNC Vec4f reference_ray(const Volume &volume, cut::Ray ray, float znear, const TransferFunc &tf,
                         const SingleScattering &ss, const ReferenceParameters &ref_params)
{
    float t_near, t_far;
    auto hit = intersect_AABB(ray, volume.domain.min, volume.domain.max, t_near, t_far);
    t_near = cut::max(t_near, znear);

    float world_step_size = ref_params.step_size;

    if (hit && (t_far > 0))
    {
        Vec3f L(0.0f);
        float t = t_near;
        float alpha = 0.0f;
        float T = 1.0f;
        float tau = 0.0f;

        float fs = map_density(volume.sample(ray.origin + t * ray.dir), ref_params);
        float bs;

        while (t < t_far)
        {
            t += world_step_size;

            auto s = volume.sample(ray.origin + t * ray.dir);
            bs = map_density(s, ref_params);

            Vec4f rgba = tf.sample(fs, bs);
            if (s >= 1e10f) // INVALID
                rgba = Vec4f(0, 1, 0, 0.05f);

            float sigma_t = rgba.w;

            Vec3f Le = ref_params.medium_emission * Vec3f(rgba.x, rgba.y, rgba.z);

            Vec3f Ls = Vec3f(0.0f);
            float sigma_s = ref_params.medium_scattering;
            if (ref_params.ss.enable && sigma_s > 0.0f)
            {
                Vec3f light_dir(ref_params.ss.direction_x, ref_params.ss.direction_y, ref_params.ss.direction_z);

                cut::Ray ss_ray;
                ss_ray.origin = ray.origin + t * ray.dir;
                ss_ray.dir = -light_dir;
                auto T_ss = compute_transmittance(volume, tf, ss, ss_ray, ref_params);

                float pf = henyey_greenstein(dot(-ray.dir, light_dir), ref_params.ss.henyey_greenstein_g);
                Ls = T_ss * pf * Vec3f(ref_params.ss.intensity_r, ref_params.ss.intensity_g, ref_params.ss.intensity_b);
            }

            // compositing with Preintegration(!)
            L += (Le + Ls * sigma_s) * T;
            tau += sigma_t;
            T = exp(-tau);
            alpha = 1.0f - T;

            fs = bs;

            if (alpha >= ref_params.early_ray_termination)
            {
                alpha = 1.f;
                break;
            }
        }

        return Vec4f(L, alpha);
    }
    else
        return Vec4f(0.0f);
}

#endif // MRAY_RAYMARCH_REFERENCE_H
