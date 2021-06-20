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
#include "moment_raymarcher.h"

#ifdef CUDA_SUPPORT
#include "moment_raymarcher.cuh"
#endif

#include "cut/raytracing.h"
#include "raymarch_reference.h"
#include "raymarch_generate.h"
#include "raymarch_rayhistogram.h"
#include "raymarch_generate_samples.h"
#include "raymarch_reconstruct.h"
#include "raymarch_single_scattering_cache.h"
#include "raymarch_resample.h"
#include "ray_histogram_sampler.h"
#include "moment_compaction.h"
#include "single_scattering_cache.h"
#include "scatter_grid.h"
#include "cut/timing.h"

using namespace cut;

template <typename Volume, typename SingleScattering>
void reference_host(const cut::Camera &cam, const Volume &volume, const TransferFunction &tf, ColorImageHost &img,
                    const SingleScattering &ss, const ReferenceParameters &params)
{
    SCOPED_CPU_QUERY("Raymarching");
#pragma omp parallel for default(none) shared(cam, volume, tf, ss, img, params)
    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            Vec2f ndc(x / static_cast<float>(img.width) * 2.f - 1.f, y / static_cast<float>(img.height) * 2.f - 1.f);

            img.set_color(x, y, reference_ray(volume, get_eye_ray(cam, ndc), cam.znear, tf, ss, params));
        }
    }
}

void generate_reference(cut::Camera cam, const TransferFunction &tf, const Particles &volume, ColorImageHost &img,
                        const ReferenceParameters &params)
{
#ifdef CUDA_SUPPORT
    generate_reference_device(cam, volume, tf, img, params);
#else
    reference_host(cam, volume, tf, img, SingleScatteringBruteForce{}, params);
#endif
}

void generate_reference(cut::Camera cam, const TransferFunction &tf, const RegularGrid &volume, ColorImageHost &img,
                        const ReferenceParameters &params)
{
#ifdef CUDA_SUPPORT
    generate_reference_device(cam, volume, tf, img, params);
#else
    reference_host(cam, volume, tf, img, SingleScatteringBruteForce{}, params);
#endif
}

template <typename Volume>
void generate_samples_host(const cut::Camera &cam, const Volume &volume, SamplesImageHost &img,
                           const GenerationParameters &params)
{
#pragma omp parallel for default(none) shared(cam, volume, img, params)
    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            Vec2f ndc(x / static_cast<float>(img.width) * 2.f - 1.f, y / static_cast<float>(img.height) * 2.f - 1.f);
            auto idx = img.get_idx(x, y);

            generate_samples_ray(volume, get_eye_ray(cam, ndc), cam.znear, SamplesWriter(&img.data[idx]), params);
        }
    }
}

void generate_samples(cut::Camera cam, const Particles &volume, SamplesImageHost &img,
                      const GenerationParameters &params)
{
#ifdef CUDA_SUPPORT
    generate_samples_device(cam, volume, img, params);
#else
    generate_samples_host(cam, volume, img, params);
#endif
}

void generate_samples(cut::Camera cam, const RegularGrid &volume, SamplesImageHost &img,
                      const GenerationParameters &params)
{
#ifdef CUDA_SUPPORT
    generate_samples_device(cam, volume, img, params);
#else
    generate_samples_host(cam, volume, img, params);
#endif
}

template <typename Volume>
void generate_host(const cut::Camera &cam, const Volume &volume, MomentImageHost &img,
                   const GenerationParameters &params)
{
    img.domain = volume.domain;

    SCOPED_CPU_QUERY("Generate");
#pragma omp parallel for default(none) shared(cam, volume, img, params)
    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            Vec2f ndc(x / static_cast<float>(img.width) * 2.f - 1.f, y / static_cast<float>(img.height) * 2.f - 1.f);
            auto idx = img.get_idx(x, y);

            {
                vector<float> trig_moments(img.num_moments);
                for (size_t i = 0; i < trig_moments.size(); ++i)
                    trig_moments[i] = 0.0f;

                auto b = generate_ray(volume, get_eye_ray(cam, ndc), cam.znear, trig_moments.data(), &img.data[idx],
                                      img.num_moments, params);

                if (img.has_bounds())
                    img.bounds[y * img.width + x] = b;
            }

            if (params.compact_image)
            {
                vector<float_complex> temp0(img.num_moments);
                vector<float_complex> temp1(img.num_moments);
                vector<float_complex> temp2(img.num_moments);
                compute_num_moments(img, x, y, params.error_threshold, temp0.data(), temp1.data(), temp2.data());
            }
        }
    }

    if (params.compact_image)
        img.compact();

    if (params.prediction_coding)
    {
        assert(params.compact_image);
        img.prediction_encode(params.coding_warp, params.bias);
    }
}

void generate(cut::Camera cam, const Particles &volume, MomentImageHost &img, const GenerationParameters &params)
{
#ifdef CUDA_SUPPORT
    generate_device(cam, volume, img, params);
#else
    generate_host(cam, volume, img, params);
#endif
}
void generate(cut::Camera cam, const RegularGrid &volume, MomentImageHost &img, const GenerationParameters &params)
{
#ifdef CUDA_SUPPORT
    generate_device(cam, volume, img, params);
#else
    generate_host(cam, volume, img, params);
#endif
}

template <typename Volume>
void generate_rayhistogram_host(const cut::Camera &cam, const Volume &volume, RayHistogramImageHost &img,
                                const RayHistogramGenerationParameters &params)
{
    vector<PixelRayHistogram> pixel_hists(img.height * img.width);

#pragma omp parallel for default(none) shared(pixel_hists, cam, volume, img, params)
    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            Vec2f ndc(x / static_cast<float>(img.width) * 2.f - 1.f, y / static_cast<float>(img.height) * 2.f - 1.f);

            generate_rayhistogram_ray(volume, cut::get_eye_ray(cam, ndc), cam.znear, &pixel_hists[y * img.width + x],
                                      params);
        }
    }

    img.create_from(pixel_hists);
}

void generate_rayhistogram(cut::Camera cam, const Particles &volume, RayHistogramImageHost &img,
                           const RayHistogramGenerationParameters &params)
{
    generate_rayhistogram_host(cam, volume, img, params);
}
void generate_rayhistogram(cut::Camera cam, const RegularGrid &volume, RayHistogramImageHost &img,
                           const RayHistogramGenerationParameters &params)
{
    generate_rayhistogram_host(cam, volume, img, params);
}

SingleScatteringCacheHost reconstruct_single_scattering_cache(const SingleScatteringImageHost &ss,
                                                              const TransferFunction &tf,
                                                              const ReconstructionParameters &params)
{
    SCOPED_CPU_QUERY("Single-scattering cache");

    ScatterGridHost grid(Vec3i(params.ss.cache_size_x, params.ss.cache_size_y, params.ss.cache_size_z), ss.img.domain,
                         0.f);

#pragma omp parallel for default(none) shared(ss, tf, params, grid)
    for (int y = 0; y < ss.img.height; ++y)
    {
        for (int x = 0; x < ss.img.width; ++x)
        {
            auto mi = ss.img;
            Vec2f ndc(x / static_cast<float>(mi.width) * 2.f - 1.f, y / static_cast<float>(mi.height) * 2.f - 1.f);
            auto num_moments = mi.get_num_moments(x, y);

            if (num_moments == 0)
                continue;

            if (params.use_truncated_fourier)
            {
                vector<float> pmoments(num_moments);
                mi.get_trig_moments(x, y, pmoments);

                TruncatedFourierReconstructionDefault rec(pmoments.size(), pmoments.data(), mi.get_bounds(x, y));
                reconstruct_single_scattering_ray(get_eye_ray(ss.cam, ndc), mi.domain, ss.cam.znear, rec, tf, grid,
                                                  params);
            }
            else
            {
                auto pmoments = mi.prepare_moments(x, y, params.bias);
                MESEReconstructionDefault rec(pmoments.size(), pmoments.data(), mi.get_bounds(x, y));
                reconstruct_single_scattering_ray(get_eye_ray(ss.cam, ndc), mi.domain, ss.cam.znear, rec, tf, grid,
                                                  params);
            }
        }
    }

    return SingleScatteringCacheHost(grid.to_volume());
}

void generate_reference_reconstructed_ss(cut::Camera cam, const TransferFunction &tf, const Particles &volume,
                                         ColorImageHost &img, const SingleScatteringImageHost &ss,
                                         const ReconstructionParameters &params)
{
#ifdef CUDA_SUPPORT
    generate_reference_reconstructed_ss_device(cam, tf, volume, img, ss, params);
#else
    auto cache = reconstruct_single_scattering_cache(ss, tf, params);
    reference_host(cam, volume, tf, img, cache, ref_from_rec_params(params));
#endif
}
void generate_reference_reconstructed_ss(cut::Camera cam, const TransferFunction &tf, const RegularGrid &volume,
                                         ColorImageHost &img, const SingleScatteringImageHost &ss,
                                         const ReconstructionParameters &params)
{
#ifdef CUDA_SUPPORT
    generate_reference_reconstructed_ss_device(cam, tf, volume, img, ss, params);
#else
    auto cache = reconstruct_single_scattering_cache(ss, tf, params);
    reference_host(cam, volume, tf, img, cache, ref_from_rec_params(params));
#endif
}

template <typename SingleScattering>
void reconstruct_host(const cut::Camera &cam, const MomentImageHost &mi, const TransferFunction &tf,
                      ColorImageHost &cimg, SingleScattering ss, const ReconstructionParameters &params)
{
    {
        SCOPED_CPU_QUERY("Preparing single-scattering");
        ss.prepare(params.bias);
    }

    SCOPED_CPU_QUERY("Reconstruct");
#pragma omp parallel for default(none) shared(cam, mi, cimg, tf, ss, params)
    for (int y = 0; y < mi.height; ++y)
    {
        for (int x = 0; x < mi.width; ++x)
        {
            Vec2f ndc(x / static_cast<float>(mi.width) * 2.f - 1.f, y / static_cast<float>(mi.height) * 2.f - 1.f);
            auto num_moments = mi.get_num_moments(x, y);

            if (num_moments == 0)
            {
                cimg.set_color(x, y, reconstruct_zero_ray(tf));
                continue;
            }

            Vec4f color(0.0f);
            if (params.use_truncated_fourier)
            {
                vector<float> pmoments(num_moments);
                mi.get_trig_moments(x, y, pmoments);

                TruncatedFourierReconstructionDefault rec(pmoments.size(), pmoments.data(), mi.get_bounds(x, y));
                color = reconstruct_ray(get_eye_ray(cam, ndc), mi.domain, cam.znear, rec, tf, ss, params);
            }
            else
            {
                auto pmoments = mi.prepare_moments(x, y, params.bias);

                MESEReconstructionDefault rec(pmoments.size(), pmoments.data(), mi.get_bounds(x, y));
                color = reconstruct_ray(get_eye_ray(cam, ndc), mi.domain, cam.znear, rec, tf, ss, params);
            }

            cimg.set_color(x, y, color);
        }
    }
}

void reconstruct(cut::Camera cam, const MomentImageHost &mi, const TransferFunction &tf, ColorImageHost &cimg,
                 const ReconstructionParameters &params)
{
#ifdef CUDA_SUPPORT
    reconstruct_device(cam, mi, tf, cimg, params);
#else
    reconstruct_host(cam, mi, tf, cimg, NoSingleScattering{}, params);
#endif
}

void reconstruct(cut::Camera cam, const MomentImageHost &mi, const TransferFunction &tf, ColorImageHost &cimg,
                 const SingleScatteringImageHost &ss, const ReconstructionParameters &params)
{
#ifdef CUDA_SUPPORT
    reconstruct_device(cam, mi, tf, cimg, ss, params);
#else
    if (params.ss.use_cache)
    {
        auto cache = reconstruct_single_scattering_cache(ss, tf, params);
        reconstruct_host(cam, mi, tf, cimg, cache, params);
    }
    else
        reconstruct_host(cam, mi, tf, cimg, ss, params);
#endif
}

RegularGrid resample_volume(cut::Camera cam, const MomentImageHost &mi,
                            const ResamplingReconstructionParameters &params)
{
    SCOPED_CPU_QUERY("Resampling");

    ScatterGridHost grid(Vec3i(params.res_x, params.res_y, params.res_z), mi.domain, INVALID_DENSITY);

#pragma omp parallel for default(none) shared(mi, cam, params, grid)
    for (int y = 0; y < mi.height; ++y)
    {
        for (int x = 0; x < mi.width; ++x)
        {
            Vec2f ndc(x / static_cast<float>(mi.width) * 2.f - 1.f, y / static_cast<float>(mi.height) * 2.f - 1.f);
            auto num_moments = mi.get_num_moments(x, y);

            if (num_moments == 0)
                continue;

            if (params.rec.use_truncated_fourier)
            {
                vector<float> pmoments(num_moments);
                mi.get_trig_moments(x, y, pmoments);

                TruncatedFourierReconstructionDefault rec(pmoments.size(), pmoments.data(), mi.get_bounds(x, y));
                reconstruct_resample_ray(get_eye_ray(cam, ndc), mi.domain, cam.znear, rec, grid, params.rec);
            }
            else
            {
                auto pmoments = mi.prepare_moments(x, y, params.rec.bias);
                MESEReconstructionDefault rec(pmoments.size(), pmoments.data(), mi.get_bounds(x, y));
                reconstruct_resample_ray(get_eye_ray(cam, ndc), mi.domain, cam.znear, rec, grid, params.rec);
            }
        }
    }

    return grid.to_volume();
}

void reconstruct_reprojected_host(const cut::Camera &cam, const MomentImageHost &mi, const TransferFunction &tf,
                                  ColorImageHost &cimg, const cut::Camera &mi_cam,
                                  const ReconstructionParameters &params)
{
    vector<float> pdata(mi.data.size());
#pragma omp parallel for default(none) shared(pdata, mi, params)
    for (int y = 0; y < mi.height; ++y)
    {
        for (int x = 0; x < mi.width; ++x)
        {
            auto num_moments = mi.get_num_moments(x, y);
            if (num_moments == 0)
                continue;

            vector<float> pmoments;
            if (params.use_truncated_fourier)
            {
                pmoments.resize(num_moments);
                mi.get_trig_moments(x, y, pmoments);
            }
            else
            {
                pmoments = mi.prepare_moments(x, y, params.bias);
            }
            std::copy(pmoments.begin(), pmoments.end(), &pdata[mi.get_idx(x, y)]);
        }
    }

    MomentImageHost pmi = mi;
    pmi.data = std::move(pdata);

    SCOPED_CPU_QUERY("Reconstruct");
#pragma omp parallel for default(none) shared(cam, pmi, mi_cam, cimg, tf, params)
    for (int y = 0; y < cimg.height; ++y)
    {
        for (int x = 0; x < cimg.width; ++x)
        {
            Vec2f ndc(x / static_cast<float>(cimg.width) * 2.f - 1.f, y / static_cast<float>(cimg.height) * 2.f - 1.f);

            vector<float> tmp(pmi.num_moments);
            Vec4f color = reconstruct_reprojected_ray(get_eye_ray(cam, ndc), pmi, mi_cam, cam.znear, tf, tmp.data(), params);
            cimg.set_color(x, y, color);
        }
    }
}

void reconstruct_resampled(cut::Camera cam, const MomentImageHost &mi, const TransferFunction &tf, ColorImageHost &cimg,
                           const vector<cut::Camera> &new_cam, const string &output, Vec3f background,
                           const ResamplingReconstructionParameters &params)
{
#ifdef CUDA_SUPPORT
    reconstruct_resampled_device(cam, mi, tf, cimg, new_cam, output, background, params);
#else
    if (params.use_cache)
    {
        auto volume = resample_volume(cam, mi, params);

        for (size_t i = 0; i < new_cam.size(); ++i)
        {
            reference_host(new_cam[i], volume, tf, cimg, SingleScatteringBruteForce{}, ref_from_rec_params(params.rec));
            write_PPM(output + "_" + std::to_string(i) + ".ppm", cimg, background);
        }
    }
    else
    {
        for (size_t i = 0; i < new_cam.size(); ++i)
        {
            reconstruct_reprojected_host(new_cam[i], mi, tf, cimg, cam, params.rec);
            write_PPM(output + "_" + std::to_string(i) + ".ppm", cimg, background);
        }
    }
#endif
}

void reconstruct_samples_host(const cut::Camera &cam, const MomentImageHost &mi, SamplesImageHost &img,
                              const ReconstructionParameters &params)
{
#pragma omp parallel for default(none) shared(cam, mi, img, params)
    for (int y = 0; y < mi.height; ++y)
    {
        for (int x = 0; x < mi.width; ++x)
        {
            Vec2f ndc(x / static_cast<float>(mi.width) * 2.f - 1.f, y / static_cast<float>(mi.height) * 2.f - 1.f);
            auto s_idx = img.get_idx(x, y);
            auto num_moments = mi.get_num_moments(x, y);

            if (num_moments == 0)
            {
                img.data[s_idx] = INVALID_SAMPLE;
                continue;
            }

            if (params.use_truncated_fourier)
            {
                vector<float> pmoments(num_moments);
                mi.get_trig_moments(x, y, pmoments);

                TruncatedFourierReconstructionDefault rec(pmoments.size(), pmoments.data(), mi.get_bounds(x, y));
                reconstruct_samples_ray(get_eye_ray(cam, ndc), mi.domain, cam.znear, rec,
                                        SamplesWriter(&img.data[s_idx]), params);
            }
            else
            {
                auto pmoments = mi.prepare_moments(x, y, params.bias);

                MESEReconstructionDefault rec(pmoments.size(), pmoments.data(), mi.get_bounds(x, y));
                reconstruct_samples_ray(get_eye_ray(cam, ndc), mi.domain, cam.znear, rec,
                                        SamplesWriter(&img.data[s_idx]), params);
            }
        }
    }
}

void reconstruct_samples(cut::Camera cam, const MomentImageHost &mi, SamplesImageHost &img,
                         const ReconstructionParameters &params)
{
#ifdef CUDA_SUPPORT
    reconstruct_samples_device(cam, mi, img, params);
#else
    reconstruct_samples_host(cam, mi, img, params);
#endif
}

template <typename Volume>
void reconstruct_error_host(const cut::Camera &cam, const Volume &volume, MomentImageHost &img,
                            const ErrorReconstructionParameters &params)
{
#pragma omp parallel for default(none) shared(cam, volume, img, params)
    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            Vec2f ndc(x / static_cast<float>(img.width) * 2.f - 1.f, y / static_cast<float>(img.height) * 2.f - 1.f);
            auto num_moments = img.get_num_moments(x, y);

            vector<float> pmoments(num_moments);

            if (num_moments > 0)
                pmoments = img.prepare_moments(x, y, params.bias);

            MESEReconstructionDefault rec(pmoments.size(), pmoments.data(), img.get_bounds(x, y));
            float e = reconstruct_error(volume, get_eye_ray(cam, ndc), cam.znear, rec, params);

            img.set_error_bound(x, y, e);
        }
    }
}

void reconstruct_error(cut::Camera cam, const Particles &volume, MomentImageHost &mi,
                       const ErrorReconstructionParameters &params)
{
    assert(mi.has_error_bounds());
#ifdef CUDA_SUPPORT
    reconstruct_error_device(cam, volume, mi, params);
#else
    reconstruct_error_host(cam, volume, mi, params);
#endif
}

void reconstruct_error(cut::Camera cam, const RegularGrid &volume, MomentImageHost &mi,
                       const ErrorReconstructionParameters &params)
{
    assert(mi.has_error_bounds());
#ifdef CUDA_SUPPORT
    reconstruct_error_device(cam, volume, mi, params);
#else
    reconstruct_error_host(cam, volume, mi, params);
#endif
}

template <typename SingleScattering>
void reconstruct_uncertainty_interpolation_host(const cut::Camera &cam, const MomentImageHost &mi,
                                                const TransferFunction &tf, ColorImageHost &cimg, SingleScattering ss,
                                                const UncertaintyReconstructionParameters &params)
{
    ss.prepare(params.rec.bias);

#pragma omp parallel for default(none) shared(cam, mi, cimg, tf, ss, params)
    for (int y = 0; y < mi.height; ++y)
    {
        for (int x = 0; x < mi.width; ++x)
        {
            Vec2f ndc(x / static_cast<float>(mi.width) * 2.f - 1.f, y / static_cast<float>(mi.height) * 2.f - 1.f);
            auto num_moments = mi.get_num_moments(x, y);

            if (num_moments == 0)
            {
                cimg.set_color(x, y, reconstruct_zero_ray(tf));
                continue;
            }

            Vec4f color(0.0f);

            if (params.rec.use_truncated_fourier)
            {
                vector<float> pmoments(num_moments);
                mi.get_trig_moments(x, y, pmoments);

                TruncatedFourierReconstructionDefault mese(pmoments.size(), pmoments.data(), mi.get_bounds(x, y));
                ErrorBoundMESEReconstruction<TruncatedFourierReconstructionDefault> bound_rec(mese,
                                                                                              mi.get_error_bound(x, y));
                ErrorBoundInterpolator<ErrorBoundMESEReconstruction<TruncatedFourierReconstructionDefault>>
                    bound_interp(bound_rec, params.bound_interpolation);

                color = reconstruct_ray(get_eye_ray(cam, ndc), mi.domain, cam.znear, bound_interp, tf, ss, params.rec);
            }
            else
            {
                auto pmoments = mi.prepare_moments(x, y, params.rec.bias);

                MESEReconstructionDefault mese(pmoments.size(), pmoments.data(), mi.get_bounds(x, y));
                ErrorBoundMESEReconstruction<MESEReconstructionDefault> bound_rec(mese, mi.get_error_bound(x, y));
                ErrorBoundInterpolator<ErrorBoundMESEReconstruction<MESEReconstructionDefault>> bound_interp(
                    bound_rec, params.bound_interpolation);

                color = reconstruct_ray(get_eye_ray(cam, ndc), mi.domain, cam.znear, bound_interp, tf, ss, params.rec);
            }

            cimg.set_color(x, y, color);
        }
    }
}

void reconstruct_uncertainty_interpolation(cut::Camera cam, const MomentImageHost &mi, const TransferFunction &tf,
                                           ColorImageHost &cimg, const UncertaintyReconstructionParameters &params)
{
#ifdef CUDA_SUPPORT
    reconstruct_uncertainty_interpolation_device(cam, mi, tf, cimg, params);
#else
    reconstruct_uncertainty_interpolation_host(cam, mi, tf, cimg, NoSingleScattering{}, params);
#endif
}

void reconstruct_uncertainty_interpolation(cut::Camera cam, const MomentImageHost &mi, const TransferFunction &tf,
                                           ColorImageHost &cimg, const SingleScatteringImageHost &ss,
                                           const UncertaintyReconstructionParameters &params)
{
#ifdef CUDA_SUPPORT
    reconstruct_uncertainty_interpolation_device(cam, mi, tf, cimg, ss, params);
#else
    reconstruct_uncertainty_interpolation_host(cam, mi, tf, cimg, ss, params);
#endif
}

template <typename SingleScattering>
void reconstruct_uncertainty_convolution_host(const cut::Camera &cam, const MomentImageHost &mi,
                                              const TransferFunction &tf, ColorImageHost &cimg, SingleScattering ss,
                                              const UncertaintyReconstructionParameters &params)
{
    ss.prepare(params.rec.bias);

#pragma omp parallel for default(none) shared(cam, mi, cimg, tf, ss, params)
    for (int y = 0; y < mi.height; ++y)
    {
        for (int x = 0; x < mi.width; ++x)
        {
            Vec2f ndc(x / static_cast<float>(mi.width) * 2.f - 1.f, y / static_cast<float>(mi.height) * 2.f - 1.f);
            auto num_moments = mi.get_num_moments(x, y);

            if (num_moments == 0)
            {
                cimg.set_color(x, y, reconstruct_zero_ray(tf));
                continue;
            }

            Vec4f color(0.0f);
            if (params.rec.use_truncated_fourier)
            {
                vector<float> pmoments(num_moments);
                mi.get_trig_moments(x, y, pmoments);
                TruncatedFourierReconstructionDefault mese(pmoments.size(), pmoments.data(), mi.get_bounds(x, y));

                if (params.error_type == MAX_ERROR)
                {
                    ErrorBoundMESEReconstruction<TruncatedFourierReconstructionDefault> bound_rec(
                        mese, mi.get_error_bound(x, y));
                    color = reconstruct_ray_uncertainty_tf(get_eye_ray(cam, ndc), mi.domain, cam.znear, bound_rec, tf,
                                                           ss, params.rec);
                }
                else
                {
                    MeanStdDevMESEReconstruction<TruncatedFourierReconstructionDefault> bound_rec(
                        mese, mi.get_error_bound(x, y));
                    color = reconstruct_ray_uncertainty_tf(get_eye_ray(cam, ndc), mi.domain, cam.znear, bound_rec, tf,
                                                           ss, params.rec);
                }
            }
            else
            {
                auto pmoments = mi.prepare_moments(x, y, params.rec.bias);
                MESEReconstructionDefault mese(pmoments.size(), pmoments.data(), mi.get_bounds(x, y));

                if (params.error_type == MAX_ERROR)
                {
                    ErrorBoundMESEReconstruction<MESEReconstructionDefault> bound_rec(mese, mi.get_error_bound(x, y));
                    color = reconstruct_ray_uncertainty_tf(get_eye_ray(cam, ndc), mi.domain, cam.znear, bound_rec, tf,
                                                           ss, params.rec);
                }
                else
                {
                    MeanStdDevMESEReconstruction<MESEReconstructionDefault> bound_rec(mese, mi.get_error_bound(x, y));
                    color = reconstruct_ray_uncertainty_tf(get_eye_ray(cam, ndc), mi.domain, cam.znear, bound_rec, tf,
                                                           ss, params.rec);
                }
            }

            cimg.set_color(x, y, color);
        }
    }
}

void reconstruct_uncertainty_convolution(cut::Camera cam, const MomentImageHost &mi, const TransferFunction &tf,
                                         ColorImageHost &cimg, const UncertaintyReconstructionParameters &params)
{
#ifdef CUDA_SUPPORT
    reconstruct_uncertainty_convolution_device(cam, mi, tf, cimg, params);
#else
    reconstruct_uncertainty_convolution_host(cam, mi, tf, cimg, NoSingleScattering{}, params);
#endif
}

void reconstruct_rayhistogram_host(const cut::Camera &cam, const RayHistogramImageHost &img, const TransferFunction &tf,
                                   ColorImageHost &cimg, const RayHistogramReconstructionParameters &params)
{
    SCOPED_CPU_QUERY("Sample ray-histograms");
#pragma omp parallel for default(none) shared(cam, img, cimg, tf, params)
    for (int y = 0; y < img.height; ++y)
    {
        SimpleRayHistogramSamplerHost sampler;

        for (int x = 0; x < img.width; ++x)
        {
            Vec2f ndc(x / static_cast<float>(img.width) * 2.f - 1.f, y / static_cast<float>(img.height) * 2.f - 1.f);

            auto color = reconstruct_rayhistogram_ray(cut::get_eye_ray(cam, ndc), img.domain, cam.znear,
                                                      img.get_view(x, y), sampler, tf, params);

            cimg.set_color(x, y, color);
        }
    }
}

void reconstruct_rayhistogram(cut::Camera cam, const RayHistogramImageHost &img, const TransferFunction &tf,
                              ColorImageHost &cimg, const RayHistogramReconstructionParameters &params)
{
#ifdef CUDA_SUPPORT
    reconstruct_rayhistogram_device(cam, img, tf, cimg, params);
#else
    reconstruct_rayhistogram_host(cam, img, tf, cimg, params);
#endif
}

void reconstruct_rayhistogram_samples_host(const cut::Camera &cam, const RayHistogramImageHost &img,
                                           SamplesImageHost &simg, const RayHistogramReconstructionParameters &params)
{
#pragma omp parallel for default(none) shared(cam, img, simg, params)
    for (int y = 0; y < img.height; ++y)
    {
        SimpleRayHistogramSamplerHost sampler;

        for (int x = 0; x < img.width; ++x)
        {
            Vec2f ndc(x / static_cast<float>(img.width) * 2.f - 1.f, y / static_cast<float>(img.height) * 2.f - 1.f);
            auto idx = simg.get_idx(x, y);

            reconstruct_rayhistogram_samples_ray(cut::get_eye_ray(cam, ndc), img.domain, cam.znear, img.get_view(x, y),
                                                 sampler, SamplesWriter(&simg.data[idx]), params);
        }
    }
}

void reconstruct_rayhistogram_samples(cut::Camera cam, const RayHistogramImageHost &img, SamplesImageHost &simg,
                                      const RayHistogramReconstructionParameters &params)
{
#ifdef CUDA_SUPPORT
    reconstruct_rayhistogram_samples_device(cam, img, simg, params);
#else
    reconstruct_rayhistogram_samples_host(cam, img, simg, params);
#endif
}