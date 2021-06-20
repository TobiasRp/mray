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
#ifndef MRAY_MOMENT_COMPACTION_H
#define MRAY_MOMENT_COMPACTION_H

#include "mese/complex_algebra.h"
#include "mese/MESE_dynamic.h"

/**
 * For num_moments << max_moments, computes the (unknown) moments (max_moments - num_moments).
 * Cf. Eq. (B.3) in the supplementary document from Peters et al. 2019.
 */
template <typename float_complex_array>
FUNC void evaluate_unknown_moments(int num_moments, const float_complex_array exp_moments, int max_moments,
                                   float_complex_array new_exp_moments, float_complex_array temp_levinson)
{
    auto m = max_moments - 1;
    auto n = num_moments - 1;
    assert(m >= n);

    levinsonsAlgorithm(num_moments, temp_levinson, exp_moments, new_exp_moments);
    for (int i = 0; i < n + 1; ++i)
        temp_levinson[i] = 6.28318531f * temp_levinson[i];

    for (int l = 0; l < n + 1; ++l)
        new_exp_moments[l] = exp_moments[l];

    float_complex inv_lev0 = (-1.0f / temp_levinson[0]);
    for (int k = 0; k < m - n; ++k)
    {
        new_exp_moments[n + k + 1] = float_complex(0.f);

        for (int j = 1; j < n + 1; ++j)
            new_exp_moments[n + k + 1] = new_exp_moments[n + k + 1] + new_exp_moments[j + k] * temp_levinson[n + 1 - j];

        new_exp_moments[n + k + 1] = new_exp_moments[n + k + 1] * inv_lev0;
    }
}

/**
 * Converts exponential to bounded moments, but assumes the first n coefficients are already known!
 */
template <typename float_complex_array>
FUNC void exponential_to_bounded_moments(int num_moments, const float_complex_array exp_moments,
                                         float_complex_array bounded_moments, int n)
{
    assert(n >= 1);
    auto gamma_0 = exp_moments[0] / 2.f;

    for (int l = n; l < num_moments; ++l)
    {
        bounded_moments[l] = 0.0f;
        for (int j = 0; j < l; ++j)
            bounded_moments[l] = bounded_moments[l] + (l - j) / (l * gamma_0) * exp_moments[l] * bounded_moments[l - j];
        bounded_moments[l] = exp_moments[l] / (float_complex(0.0f, 2.0f) * M_PI * gamma_0) - bounded_moments[l];
    }
}

/**
 * Computes the RMSE between the last (m-n) moments.
 */
template <typename float_complex_array>
FUNC float rmse(int n, int m, const float_complex_array pred, const float *target)
{
    float sum = 0.0f;
    for (int k = n; k < m; ++k)
    {
        auto diff = pred[k].x - target[k];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

/**
 * Computes the rRMSE on bounded trigonometric moments by estimating the (m-n) unknown moments.
 */
template <typename float_complex_array>
FUNC float evaluate_rrmse(int n, int num_moments, const float *moments, float_complex_array temp_exp,
                          float_complex_array temp_extended, float_complex_array temp_levinson, float norm)
{
    evaluate_unknown_moments(n, temp_exp, num_moments, temp_extended, temp_levinson);

    auto extended_moments = temp_levinson;
    for (int l = 0; l < n; ++l)
        extended_moments[l] = moments[l];
    exponential_to_bounded_moments(num_moments, temp_extended, extended_moments, n);

    return rmse(n, num_moments, extended_moments, moments) * norm;
}

template <typename float_complex_array>
FUNC int determine_best_number_moments(int num_moments, const float *moments, float error_threshold,
                                       float_complex_array temp_exp, float_complex_array temp_extended,
                                       float_complex_array temp_levinson)
{
    if (error_threshold == 0.0f)
        return num_moments;

    float norm = 1.0f / moments[0];
    trigonometricToExponentialMoments(num_moments, temp_exp, moments);

    int n = num_moments / 2;
    int s = num_moments / 4;

    while (s > 0)
    {
        float rrmse = evaluate_rrmse(n, num_moments, moments, temp_exp, temp_extended, temp_levinson, norm);
        if (cut::abs(rrmse - error_threshold) < 1e-8f)
            return n; // We're "exactly" at the error_threshold!
        else if (rrmse > error_threshold)
            n += s;
        else
            n -= s;

        s = s / 2;
    }

    return n;
}

/**
 * Computes/estimates the actual number of moments per pixel.
 * Writes the number to the moment image.
 */
template <typename MomentImage, typename float_complex_array>
FUNC void compute_num_moments(MomentImage &mi, int x, int y, float error_threshold, float_complex_array temp_exp,
                              float_complex_array temp_extended, float_complex_array temp_levinson)
{
    assert(!mi.is_compact);

    auto old_idx = mi.get_idx(x, y);
    bool is_zero = false;
    if (mi.has_bounds())
    {
        auto b = mi.get_bounds(x, y);
        is_zero = b.x == b.y;
    }
    else
    {
        is_zero = mi.data[old_idx] == 0.0f;
    }

    int n = 0;
    if (!is_zero)
    {
        n = determine_best_number_moments(mi.get_num_moments(x, y), &mi.data[old_idx], error_threshold, temp_exp,
                                          temp_extended, temp_levinson);
    }

    assert(n <= mi.num_moments);
    mi.index[y * mi.width + x] = n;
}

#endif // MRAY_MOMENT_COMPACTION_H
