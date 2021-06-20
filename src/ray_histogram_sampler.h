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
#ifndef MRAY_RAY_HISTOGRAM_SAMPLER_H
#define MRAY_RAY_HISTOGRAM_SAMPLER_H

#include <random>
#include "parameters.h"
#include "transfer_function.h"

FUNC float sample_bin(int bin, float x)
{
    float bin_size = 1.f / static_cast<float>(RayHistogramGenerationParameters::NUM_BINS);
    return cut::clamp((bin + x) * bin_size, 0.f, 1.f);
}

FUNC float sample_without_importance(float x, const uint16_t *bins, const uint16_t *f, int size,
                                                  float f_sum)
{
    float sum = 0.0f;
    for (int i = 0; i < size; ++i)
    {
        float p = f[i] / f_sum;

        if (sum <= x && x <= sum + p)
        {
            return sample_bin(bins[i], (x - sum) / p);
        }
        sum += p;
    }

    // This shouldn't happen, but does due to floating point accuracy issues
    // Anyway, this just means that we're in the last bin
    return sample_bin(bins[size - 1], (x - sum) / (f[size - 1] / f_sum));
}

FUNC float sample_importance(float x, const uint16_t *bins, const uint16_t *f, int size, float f_sum,
                                          const float *importance)
{
    float sum = 0.0f;
    for (int i = 0; i < size; ++i)
    {
        auto imp = importance[bins[i]];
        float p = (f[i] / f_sum) * imp;

        if (sum <= x && x <= sum + p)
        {
            return sample_bin(bins[i], (x - sum) / p);
        }
        sum += p;
    }

    // This shouldn't happen, but does due to floating point accuracy issues
    // Anyway, this just means that we're in the last bin
    return sample_bin(bins[size - 1], (x - sum) / (f[size - 1] / f_sum));
}

// Without importance sampling (Sec. 5.1)
struct SimpleRayHistogramSamplerHost
{
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist;

    SimpleRayHistogramSamplerHost()
        : dist(0.f, 1.0f)
    {
    }

    float operator()(const uint16_t *bins, const uint16_t *f, int size, float f_sum)
    {
        float x = dist(gen);

        return sample_without_importance(x, bins, f, size, f_sum);
    }
};

inline vector<float> compute_importance(const TransferFunction &tf)
{
    vector<float> importance;
    for (int b = 0; b < RayHistogramGenerationParameters::NUM_BINS; ++b)
    {
        float b_l = b / static_cast<float>(RayHistogramGenerationParameters::NUM_BINS - 1);
        float b_u = (b + 1) / static_cast<float>(RayHistogramGenerationParameters::NUM_BINS - 1);

        int num_steps = 5;
        float i = 0.0f;
        for (int s = 0; s < num_steps; ++s)
        {
            float x = i / static_cast<float>(num_steps - 1);
            i += tf.sample((1.f - x) * b_l + x * b_u, 0.0f).w;
        }
        i /= num_steps;

        importance.push_back(i);
    }

    // importance must have mean of 1
    // Otherwise we always take too few samples
    float i_sum = 0.0f;
    for (auto imp : importance)
        i_sum += imp;

    auto i_mean = i_sum / importance.size();

    for (auto &imp : importance)
        imp = imp + i_mean;

    return importance;
}

// Importance sampling (Sec. 5.1)
struct ImportanceRayHistogramSamplerHost
{
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist;

    vector<float> importance;

    ImportanceRayHistogramSamplerHost(const TransferFunction &tf)
        : dist(0.f, 1.0f)
    {
        importance = compute_importance(tf);
    }

    float operator()(const uint16_t *bins, const uint16_t *f, int size, float f_sum)
    {
        float x = dist(gen);

        return sample_importance(x, bins, f, size, f_sum, importance.data());
    }
};

#endif // MRAY_RAY_HISTOGRAM_SAMPLER_H
