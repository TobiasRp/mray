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
#ifndef MRAY_RAY_HISTOGRAM_IMAGE_H
#define MRAY_RAY_HISTOGRAM_IMAGE_H

#include "common.h"
#include "parameters.h"

// "Image and Distribution Based Volume Rendering for Large Data Sets"
// from K.C. Wang, N. Shareef, and H.W. Shen

struct SubFrustum
{
    uint16_t num_samples;
    uint32_t hist_idx;
};

// Rayhistogram for a single pixel
// Used to construct a RayHistogramImage in parallel
struct PixelRayHistogram
{
    vector<SubFrustum> frusta;
    vector<uint16_t> frequencies;
    vector<uint16_t> bin_ids;

    inline void add_frustum(uint16_t num_samples)
    {
        uint32_t hist_idx = frequencies.size();
        frusta.emplace_back(SubFrustum{num_samples, hist_idx});
    }

    inline void add_bin(uint16_t id, uint16_t f)
    {
        bin_ids.push_back(id);
        frequencies.push_back(f);
    }
};

struct PixelRayHistogramView
{
    const SubFrustum *frusta;
    const uint16_t *frequencies;
    const uint16_t *bin_ids;

    uint32_t num_frusta;
    uint32_t frequency_sum;

    FUNC PixelRayHistogramView(const uint32_t *indices, const SubFrustum *img_frusta, const uint16_t *fs,
                          const uint16_t *ids, uint32_t offset)
    {
        auto frust_idx = indices[offset];
        num_frusta = indices[offset + 1] - frust_idx;
        frusta = &img_frusta[frust_idx];
        frequencies = fs;
        bin_ids = ids;

        frequency_sum = 0;
        for (uint32_t fidx = 0; fidx < num_frusta; ++fidx)
        {
            auto hist_idx = frusta[fidx].hist_idx;
            auto hist_size = frusta[fidx + 1].hist_idx - hist_idx;

            for (uint32_t b = 0; b < hist_size; ++b)
                frequency_sum += frequencies[hist_idx + b];
        }
    }

    FUNC int get_step_size(int frustum) const { return frusta[frustum].num_samples; }

    FUNC int get_frustum_num_samples(int frustum, int sample_count, float f_sum) const
    {
        if (frustum >= static_cast<int>(num_frusta))
            return -1;

        auto hist_idx = frusta[frustum].hist_idx;
        auto hist_size = frusta[frustum + 1].hist_idx - hist_idx;

        // Sec. 5.1
        auto frustum_samples = sample_count * f_sum / frequency_sum;

        // Sec. 5.3
        int num_zero_bins = RayHistogramGenerationParameters::NUM_BINS - hist_size;

        return cut::clamp(
            static_cast<int>(frustum_samples *
                             (num_zero_bins / static_cast<float>(RayHistogramGenerationParameters::NUM_BINS))),
            1, sample_count);
    }

    FUNC float get_frustum_sum(int frustum) const
    {
        auto hist_idx = frusta[frustum].hist_idx;
        auto hist_size = frusta[frustum + 1].hist_idx - hist_idx;

        // compute frequency sum of hist
        uint32_t FS = 0;
        for (uint32_t b = 0; b < hist_size; ++b)
            FS += frequencies[hist_idx + b];

        return FS;
    }

    FUNC void get_frustum_hist(int frustum, uint32_t &hist_idx, uint32_t &hist_size) const
    {
        hist_idx = frusta[frustum].hist_idx;
        hist_size = frusta[frustum + 1].hist_idx - hist_idx;
    }
};

struct RayHistogramImageHost
{
    int width, height;
    Range<Vec3f> domain;

    vector<uint32_t> indices;
    vector<SubFrustum> frusta;
    vector<uint16_t> frequencies;
    vector<uint16_t> bin_ids;

    RayHistogramImageHost(int w, int h)
        : width(w)
        , height(h)
    {
    }

    PixelRayHistogramView get_view(int x, int y) const
    {
        return PixelRayHistogramView(indices.data(), frusta.data(), frequencies.data(), bin_ids.data(), y * width + x);
    }

    void create_from(const vector<PixelRayHistogram> &rayhists)
    {
        assert(rayhists.size() == static_cast<size_t>(width * height));

        indices.reserve(width * height + 1);

        for (const auto &ray : rayhists)
        {
            indices.push_back(frusta.size());
            for (const auto &subfrustum : ray.frusta)
                frusta.push_back(SubFrustum{subfrustum.num_samples,
                                            subfrustum.hist_idx + static_cast<uint32_t>(frequencies.size())});

            frequencies.insert(frequencies.end(), ray.frequencies.begin(), ray.frequencies.end());
            bin_ids.insert(bin_ids.end(), ray.bin_ids.begin(), ray.bin_ids.end());
        }
        indices.push_back(static_cast<uint32_t>(frusta.size()));
        frusta.push_back(SubFrustum{0, static_cast<uint32_t>(frequencies.size())});
    }
};

extern RayHistogramImageHost read_rayhistogram_image(string filename);

extern void write_rayhistogram_image(string filename, const RayHistogramImageHost &img, bool compress);

#endif // MRAY_RAY_HISTOGRAM_IMAGE_H
