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
#include "moment_image.h"

#include "cut/timing.h"
#include "moment_preparation.h"
#include "moment_prediction_coding.h"
#include "coding_transform.h"

void MomentImageHost::get_trig_moments(int x, int y, vector<float> &out) const
{
    auto num = get_num_moments(x, y);
    if (num == 0)
        return;
    else if (num > static_cast<int>(out.size()))
        out.resize(num);

    auto moments = &data[get_idx(x, y)];
    if (prediction_code)
    {
        vector<float_complex> temp_code(num);
        vector<float_complex> exp_moments(num);
        vector<float_complex> temp_eval_poly(num);
        vector<float_complex> temp(num);

        vector<float> params;
        if (requires_coding_parameters(coding_warp))
            params = get_coding_params();

        transform_dequantization_real(num, moments, temp_code.data(), WarpParameters{coding_warp, params.data()});

        decode(num, temp_code.data(), exp_moments.data(), temp_eval_poly.data(), temp.data());

        assert(!std::isnan(exp_moments[0].x));
        exponentialToTrigonometricMoments(num, temp.data(), exp_moments.data());

        for (int l = 0; l < num; ++l)
            out[l] = temp[l].x;
    }
    else
    {
        for (int l = 0; l < num; ++l)
            out[l] = moments[l];
    }
}

void MomentImageHost::compact()
{
    assert(!is_compact && !index.empty());

    vector<float> compact_data;

    uint32_t idx_sum = 0;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            auto num = index[y * width + x];
            index[y * width + x] = idx_sum;

            auto old_idx = get_idx(x, y);
            for (uint32_t m = 0; m < num; ++m)
                compact_data.push_back(data[old_idx + m]);

            idx_sum += num;
        }
    }
    index[height * width] = idx_sum;

    std::swap(data, compact_data);
    is_compact = true;
}

void MomentImageHost::prediction_encode(int warp, float bias)
{
    SCOPED_CPU_QUERY("Prediction coding");

    prediction_code = true;
    coding_warp = warp;

#pragma omp parallel for default(none), shared(data, coding_warp, bias)
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            auto idx = get_idx(x, y);
            auto num_moments = get_num_moments(x, y);

            if (num_moments <= 0)
                continue;

            data[idx] = cut::lerp(data[idx], 0.5f, bias);
            for (int i = 1; i < num_moments; ++i)
                data[idx + i] = cut::lerp(data[idx + i], 0.0f, bias);

            vector<float_complex> exp_moments(num_moments);
            trigonometricToExponentialMoments(num_moments, exp_moments.data(), &data[idx]);

            vector<float_complex> code(num_moments);
            vector<float_complex> eval_polynom(num_moments);
            vector<float_complex> temp(num_moments);
            encode(num_moments, exp_moments.data(), code.data(), eval_polynom.data(), temp.data());

            transform_quantization_real(num_moments, code.data(), data[idx], &data[idx]);
        }
    }

    if (requires_coding_parameters(coding_warp))
    {
        if (coding_warp == CODING_WARP_DEFAULT_TRANSFORMED)
            set_coding_params(find_coding_transform_parameters(*this));
    }

    // Warping
    vector<float> params = get_coding_params();
#pragma omp parallel for default(none), shared(data, coding_warp, params)
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            auto num_moments = get_num_moments(x, y);
            if (num_moments <= 0)
                continue;

            auto idx = get_idx(x, y);

            transform_quantization_real_warp(num_moments, &data[idx], WarpParameters{coding_warp, params.data()});
        }
    }
}

vector<float> MomentImageHost::prepare_moments(int x, int y, float bias) const
{
    auto idx = get_idx(x, y);
    auto num_moments = get_num_moments(x, y);

    vector<float> pmoments(num_moments);
    const float *moments = &data[idx];

    vector<float_complex> temp_exp_moments(num_moments);
    vector<float_complex> temp_eval_poly(num_moments);
    vector<float_complex> temp(num_moments);

    auto params = get_coding_params();
    if (prediction_code)
    {
        vector<float_complex> temp_code(num_moments);

        prepare_moments_from_pred_coding(num_moments, moments, pmoments.data(), temp_code.data(),
                                         temp_exp_moments.data(), temp_eval_poly.data(), temp.data(),
                                         WarpParameters{coding_warp, params.data()});
    }
    else
    {
        vector<float> moments_copy(moments, moments + num_moments);

        prepare_moments_from_std_coding(num_moments, moments_copy.data(), pmoments.data(), temp_exp_moments.data(),
                                        temp_eval_poly.data(), temp.data(), bias);
    }
    return pmoments;
}
