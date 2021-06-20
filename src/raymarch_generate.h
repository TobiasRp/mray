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
#ifndef MRAY_RAYMARCH_GENERATE_H
#define MRAY_RAYMARCH_GENERATE_H

#include "raymarch_common.h"

#define M_INV_PI 0.3183098861837907f

/**
 * Iteratively compute bounded trigonometric moments using linear interpolation.
 */
template <typename float_array>
FUNC void update_moments(int num_moments, float_array trig_moments, float p_next, float phase_step_size, float fs,
                         float bs)
{
    float p_prev = p_next - phase_step_size;

    float gradient = (bs - fs) / phase_step_size;
    float yintercept = fs - gradient * p_prev;

    for (int j = 1; j < num_moments; ++j)
    {
        float jSq = float(j) * float(j);

        auto commonSummands = float_complex(gradient / jSq, yintercept / float(j));

        trig_moments[j] +=
            ((commonSummands + float_complex(0, gradient * j * p_next / jSq)) * euler_exp(-j * p_next)).x;
        trig_moments[j] -=
            ((commonSummands + float_complex(0, gradient * j * p_prev / jSq)) * euler_exp(-j * p_prev)).x;
    }

    trig_moments[0] += 0.5 * gradient * (p_next * p_next) + yintercept * p_next;
    trig_moments[0] -= 0.5 * gradient * (p_prev * p_prev) + yintercept * p_prev;
}

FUNC void update_bound(Vec2f &bound, float new_density)
{
    bound.x = cut::min(bound.x, new_density);
    bound.y = cut::max(bound.y, new_density);
}

/**
 * Ray march to compute bounded trigonometric moments.
 */
template <typename Volume, typename float_array>
FUNC DensityBound generate_ray(const Volume &volume, cut::Ray ray, float znear, float_array trig_moments,
                               float *moments, int num_moments, const GenerationParameters &gen_params)
{
    float t_near, t_far;
    auto hit = intersect_AABB(ray, volume.domain.min, volume.domain.max, t_near, t_far);
    t_near = cut::max(t_near, znear);

    if (hit && (t_far > 0))
    {
        auto trange = cut::Vec2f(t_near, t_far);
        float world_step_size = gen_params.step_size;

        Vec2f bound(1.f, 0.f);

        float t = trange.x;
        float pi_trange_inv = M_PI / (t_far - t_near);
        float p = -M_PI;
        float phase_step_size = get_phase_step(trange.x, trange.y, world_step_size);

        float fs = map_density(volume.sample(ray.origin + t * ray.dir), gen_params);
        update_bound(bound, fs);

        float bs;

        while (t + world_step_size < trange.y)
        {
            t += world_step_size;

            // Incrementing p by phase_step_size has lead to numerical issues
            p = (t - trange.x) * pi_trange_inv - M_PI;

            bs = map_density(volume.sample(ray.origin + t * ray.dir), gen_params);
            update_bound(bound, bs);

            update_moments(num_moments, trig_moments, p, phase_step_size, fs, bs);
            fs = bs;
        }

        // Add last sample exactly at t_far, i.e. p = 0.0
        float last_step = get_max_phase() - p;

        bs = map_density(volume.sample(ray.origin + t_far * ray.dir), gen_params);
        update_bound(bound, bs);

        update_moments(num_moments, trig_moments, get_max_phase(), last_step, fs, bs);

        // Mirror signal, take only real part. Optionally transform to bounds
        if (gen_params.transform_to_bounds)
        {
            bound = Vec2f(cut::max(bound.x - gen_params.transform_bound_eps, 0.0f),
                          cut::min(bound.y + gen_params.transform_bound_eps, 1.0f));

            auto inv_relaxed_bound = 1.0 / (bound.y - bound.x);

            moments[0] = (trig_moments[0] * M_INV_PI - bound.x) * inv_relaxed_bound;
            for (int j = 1; j < num_moments; ++j)
                moments[j] = (trig_moments[j] * M_INV_PI) * inv_relaxed_bound;
        }
        else
        {
            for (int j = 0; j < num_moments; ++j)
                moments[j] = trig_moments[j] * M_INV_PI;
        }

        assert(!std::isnan(moments[0]));
        return DensityBound(bound.x, bound.y);
    }
    else if (gen_params.transform_to_bounds && gen_params.compact_image)
    {
        // Don't write zero moments, the moment compaction will interpret the moments as zero and remove them anyway!
        return DensityBound(0, 0);
    }
    else
    {
        for (int j = 0; j < num_moments; ++j)
            moments[j] = 0.f;
        return DensityBound(0, 0);
    }
}

#endif // MRAY_RAYMARCH_GENERATE_H
