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
#ifndef MRAY_RAYMARCH_RAYHISTOGRAM_H
#define MRAY_RAYMARCH_RAYHISTOGRAM_H

#include "cut/raytracing.h"
#include "cut/entropy.h"
#include "parameters.h"
#include "samples_image.h"

/* Implementation of "Image and Distribution Based Volume Rendering for Large Data Sets"
 * from K.C. Wang, N. Shareef, and H.W. Shen
 */

FUNC void reset(uint16_t *hist)
{
    for (int b = 0; b < RayHistogramGenerationParameters::NUM_BINS; ++b)
        hist[b] = 0;
}

FUNC int get_bin(float v, float data_min, float data_max)
{
    return cut::clamp(
        static_cast<int>((v - data_min) / (data_max - data_min) * (RayHistogramGenerationParameters::NUM_BINS - 1)), 0,
        RayHistogramGenerationParameters::NUM_BINS - 1);
}

FUNC float update(float e, uint16_t *hist, uint32_t &Sn, int b)
{
    assert(b >= 0 && b < RayHistogramGenerationParameters::NUM_BINS);

    if (Sn == 0)
    {
        hist[b] += 1; // Update hist
        Sn += 1;
        return 1.f / static_cast<float>(RayHistogramGenerationParameters::NUM_BINS);
    }
    else
    {
        UNUSED(e);
        hist[b] += 1; // Update hist
        Sn += 1;
        return entropy(hist, RayHistogramGenerationParameters::NUM_BINS);
    }

    // We did not implement the fast entropy update. This is not an issue since we're not
    // reporting runtime measurements.

    //    float fb = hist[b];
    //    int c = 1;
    //
    //    auto inv_sn = 1.0f / static_cast<float>(Sn);
    //    auto inv_sn_c = 1.f / static_cast<float>(Sn + c);
    //
    //    auto log_fb_sn = e_log2(fb * inv_sn);
    //    auto log_sn_inv_sn_c = e_log2(Sn * inv_sn_c);
    //    auto log_fb_c_inv_sn_c = e_log2((fb + c) * inv_sn_c);
    //
    //    float t0 = Sn * inv_sn_c;
    //    float t1 = -fb * inv_sn * log_fb_sn;
    //
    //    float t2 = -(Sn - fb);
    //    float t3 = -(fb + c) * inv_sn_c;
    //
    //    float l1 = t0 * (e - t1);
    //    float l2 = t2 * inv_sn_c * log_sn_inv_sn_c;
    //    float l3 = t3 * log_fb_c_inv_sn_c;
    //
    //    Sn += 1;
    //    hist[b] += c; // Update hist
    //    return l1 + l2 + l3;
}

template <typename RayHistogram> FUNC void save(RayHistogram *rayhist, uint16_t *hist, int num_samples)
{
    rayhist->add_frustum(num_samples);
    for (uint16_t b = 0; b < RayHistogramGenerationParameters::NUM_BINS; ++b)
        if (hist[b] > 0)
            rayhist->add_bin(b, hist[b]);
}

template <typename Volume, typename RayHistogram>
FUNC void generate_rayhistogram_ray(const Volume &volume, cut::Ray ray, float znear, RayHistogram *rayhist,
                                    const RayHistogramGenerationParameters &params)
{
    float t_near, t_far;
    auto hit = intersect_AABB(ray, volume.domain.min, volume.domain.max, t_near, t_far);
    t_near = cut::max(t_near, znear);

    float world_step_size = params.step_size;

    uint16_t hist[RayHistogramGenerationParameters::NUM_BINS];
    reset(hist);
    uint32_t freq_sum = 0;

    float e = 0.0f;
    if (hit && (t_far > 0))
    {
        float t = t_near;
        while (t < t_far)
        {
            auto v = volume.sample(ray.origin + t * ray.dir);

            e = update(e, hist, freq_sum, get_bin(v, params.data_min, params.data_max));

            if (e >= params.entropy_threshold)
            {
                save(rayhist, hist, freq_sum);
                e = 0.0f;
                freq_sum = 0;
                reset(hist);
            }

            t += world_step_size;
        }

        if (freq_sum > 0)
            save(rayhist, hist, freq_sum);
    }
}

template <typename TransferFunction, typename RayHistogram, typename HistogramSampler>
FUNC Vec4f reconstruct_rayhistogram_ray(cut::Ray ray, Range<Vec3f> domain, float znear, RayHistogram rayhist,
                                        HistogramSampler &sampler, const TransferFunction &tf,
                                        const RayHistogramReconstructionParameters &rec_params)
{
    UNUSED(znear);
    float t_near, t_far;
    auto hit = intersect_AABB(ray, domain.min, domain.max, t_near, t_far);

    if (hit && (t_far > 0))
    {
        Vec3f L(0.0f);
        float alpha = 0.0f;
        float T = 1.0f;
        float tau = 0.0f;

        for (uint32_t frustum_idx = 0; frustum_idx < rayhist.num_frusta; ++frustum_idx)
        {
            auto f_sum = rayhist.get_frustum_sum(frustum_idx);

            auto num_frustum_samples = rayhist.get_frustum_num_samples(frustum_idx, rec_params.num_samples, f_sum);

            uint32_t hist_idx, hist_size;
            rayhist.get_frustum_hist(frustum_idx, hist_idx, hist_size);

            float dt = rec_params.step_size * static_cast<float>(rayhist.get_step_size(frustum_idx)) /
                       static_cast<float>(num_frustum_samples);

            for (int fsidx = 0; fsidx < num_frustum_samples; ++fsidx)
            {
                auto s = sampler(&rayhist.bin_ids[hist_idx], &rayhist.frequencies[hist_idx], hist_size, f_sum);

                Vec4f rgba = tf.sample(s, 0); // Expect/use a 1D transfer function!!
                float sigma_t = rgba.w;

                Vec3f Le = rec_params.medium_emission * Vec3f(rgba.x, rgba.y, rgba.z);

                // compositing with>OUT< Preintegration!
                L += (sigma_t * Le) * T * dt;
                tau += sigma_t * dt;
                T = exp(-tau);
                alpha = 1.0f - T;

                if (alpha >= rec_params.early_ray_termination)
                {
                    alpha = 1.f;
                    break;
                }
            }
        }

        return Vec4f(L, alpha);
    }
    else
        return Vec4f(0.f);
}

template <typename RayHistogram, typename HistogramSampler>
FUNC void reconstruct_rayhistogram_samples_ray(cut::Ray ray, Range<Vec3f> domain, float znear, RayHistogram rayhist,
                                               HistogramSampler &sampler, SamplesWriter samples,
                                               const RayHistogramReconstructionParameters &rec_params)
{
    UNUSED(znear);
    float t_near, t_far;
    auto hit = intersect_AABB(ray, domain.min, domain.max, t_near, t_far);

    int sidx = 0;
    if (hit && (t_far > 0))
    {
        for (uint32_t frustum_idx = 0; frustum_idx < rayhist.num_frusta; ++frustum_idx)
        {
            auto f_sum = rayhist.get_frustum_sum(frustum_idx);

            auto num_frustum_samples = rayhist.get_frustum_num_samples(frustum_idx, rec_params.num_samples, f_sum);

            uint32_t hist_idx, hist_size;
            rayhist.get_frustum_hist(frustum_idx, hist_idx, hist_size);

            for (int fsidx = 0; fsidx < num_frustum_samples; ++fsidx)
            {
                auto s = sampler(&rayhist.bin_ids[hist_idx], &rayhist.frequencies[hist_idx], hist_size, f_sum);
                samples.write(sidx, s);
                ++sidx;

                if (sidx >= samples.get_max())
                    break;
            }
        }
    }

    if (sidx < samples.get_max())
        samples.write_invalid(sidx);
}

#endif // MRAY_RAYMARCH_RAYHISTOGRAM_H
