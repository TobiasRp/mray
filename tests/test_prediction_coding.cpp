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
#include <gtest/gtest.h>
#include "common.h"
#include "mese/MESE_dynamic.h"
#include "moment_prediction_coding.h"

TEST(test_prediction_coding, test_formulation_real)
{
    vector<float> trig_moments({0.22, 0.0f, 0.014, 0.0027, 0.004});

    int num_moments = trig_moments.size();
    vector<float_complex> exp_moments(num_moments);
    trigonometricToExponentialMoments(num_moments, exp_moments.data(), trig_moments.data());

    ASSERT_FLOAT_EQ(exp_moments[1].x, 0.0f);
    ASSERT_FLOAT_EQ(exp_moments[1].y, 0.0f);

    vector<float_complex> code(num_moments);
    vector<float_complex> eval_poly(num_moments);
    vector<float_complex> temp(num_moments);
    encode(trig_moments.size(), exp_moments.data(), code.data(), eval_poly.data(), temp.data());

    ASSERT_FLOAT_EQ(code[1].x, 0.0f);
    ASSERT_FLOAT_EQ(code[1].y, 0.0f);

    vector<float_complex> decoded_exp_moments(num_moments);
    vector<float_complex> decoded_eval_poly(num_moments);
    decode(num_moments, code.data(), decoded_exp_moments.data(), decoded_eval_poly.data(), temp.data());

    ASSERT_FLOAT_EQ(decoded_exp_moments[1].x, 0.0f);
    ASSERT_FLOAT_EQ(decoded_exp_moments[1].y, 0.0f);

    // We don't require the eval poly output from encode to be correct/complete, recompute it
    levinsonsAlgorithm(num_moments, eval_poly.data(), exp_moments.data(), temp.data());

    static constexpr float exp_eps = 1e-6f;
    for (int l = 0; l < num_moments; ++l)
    {
        ASSERT_FLOAT_EQ(eval_poly[l].x, decoded_eval_poly[l].x);
        ASSERT_FLOAT_EQ(eval_poly[l].y, decoded_eval_poly[l].y);

        ASSERT_NEAR(exp_moments[l].x, decoded_exp_moments[l].x, exp_eps);
        ASSERT_NEAR(exp_moments[l].y, decoded_exp_moments[l].y, exp_eps);
    }
}

TEST(test_prediction_coding, test_quantization_transform)
{
    int coding_warp = CODING_WARP_NONE;

    vector<float> trig_moments({0.22008814, 0.0f, 0.01400074, 0.00271224, 0.00426201});

    int num_moments = trig_moments.size();
    vector<float_complex> exp_moments(num_moments);
    trigonometricToExponentialMoments(num_moments, exp_moments.data(), trig_moments.data());

    vector<float_complex> code(num_moments);
    vector<float_complex> eval_poly(num_moments);
    vector<float_complex> temp(num_moments);
    encode(trig_moments.size(), exp_moments.data(), code.data(), eval_poly.data(), temp.data());

    vector<float> transformed(num_moments + 1);
    transform_quantization_real(num_moments, code.data(), trig_moments[0], transformed.data());
    transform_quantization_real_warp(num_moments, transformed.data(), WarpParameters{coding_warp, nullptr});

    // Now transform back
    vector<float_complex> transformed_code(num_moments);
    transform_dequantization_real(num_moments, transformed.data(), transformed_code.data(),
                                  WarpParameters{coding_warp, nullptr});

    static constexpr float tr_eps = 1e-7f;
    for (int l = 0; l < num_moments; ++l)
    {
        ASSERT_NEAR(transformed_code[l].x, code[l].x, tr_eps);
        ASSERT_NEAR(transformed_code[l].y, code[l].y, tr_eps);
    }
}