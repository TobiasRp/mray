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
#include "gtest/gtest.h"

#include "common.h"
#include "mese/MESE_dynamic.h"
#include "mese/MESE_4_10.h"

TEST(test_mese, reconstruction4)
{
    vector<float> moments({0.5f, 0.25f, 0.25f, 0.1f});
    vector<float> pmoments(4);

    prepareReflectanceSpectrumLagrange4(pmoments.data(), moments.data());

    float p = -M_PI + M_PI * 0.5f;
    float value_fixed = evaluateReflectanceSpectrumLagrange4(p, pmoments.data());
    float value_gen = evaluateMESELagrange(4, p, pmoments.data());

    ASSERT_FLOAT_EQ(value_fixed, value_gen);
}

TEST(test_mese, reconstruction10)
{
    vector<float> moments({0.22008814, 0.f, 0.01400074, 0.00271224, 0.00426201, 0.f, 0.f, 0.f, 0.f, 0.f});
    vector<float> pmoments(10);

    prepareReflectanceSpectrumLagrange10(pmoments.data(), moments.data());

    float p = -M_PI + M_PI * 0.5f;
    float value_fixed = evaluateReflectanceSpectrumLagrange10(p, pmoments.data());
    float value_gen = evaluateMESELagrange(10, p, pmoments.data());

    ASSERT_FLOAT_EQ(value_fixed, value_gen);
}

TEST(test_mese, trig_to_exponential_moments)
{
    vector<float> trig_moments({0.22008814, 0.f, 0.01400074, 0.00271224});
    vector<float_complex> fixed_exp_moments(4);
    vector<float_complex> general_exp_moments(4);

    trigonometricToExponentialMoments4(fixed_exp_moments.data(), trig_moments.data());

    trigonometricToExponentialMoments(4, general_exp_moments.data(), trig_moments.data());

    for (int i = 0; i < static_cast<int>(trig_moments.size()); ++i)
    {
        ASSERT_FLOAT_EQ(fixed_exp_moments[i].x, general_exp_moments[i].x);
        ASSERT_FLOAT_EQ(fixed_exp_moments[i].y, general_exp_moments[i].y);
    }
}

TEST(test_mese, trig_to_exponential_moments_complex)
{
    vector<float_complex> trig_moments({float_complex(0.22008814, 0.0f), float_complex(0.f),
                                        float_complex(0.01400074, 0.5f), float_complex(0.00271224, 0.1f)});
    vector<float_complex> fixed_exp_moments(4);
    vector<float_complex> general_exp_moments(4);

    trigonometricToExponentialMoments4(fixed_exp_moments.data(), trig_moments.data());

    trigonometricToExponentialMoments(4, general_exp_moments.data(), trig_moments.data());

    for (int i = 0; i < static_cast<int>(trig_moments.size()); ++i)
    {
        ASSERT_FLOAT_EQ(fixed_exp_moments[i].x, general_exp_moments[i].x);
        ASSERT_FLOAT_EQ(fixed_exp_moments[i].y, general_exp_moments[i].y);
    }
}

TEST(test_mese, exponential_to_trig_moments)
{
//    vector<float> trig_moments({0.92008814, 0.f, 0.01400074, 0.00271224});
    vector<float> trig_moments({5.2070445e-01,  7.4165657e-02, -1.3283056e-02,  1.2487045e-02,
                                1.3437086e-02, -9.5647079e-04, -1.2216626e-03,  2.1146263e-03,
                                -5.4606516e-04,  1.0199437e-02,  1.1894707e-02, -1.9188672e-02,
                                -5.5241492e-03,  2.8166445e-02,  6.7452774e-03, -1.9198583e-02,
                                4.3223342e-03, -6.1640824e-04,  7.8988671e-03,  2.4080461e-02,
                                -9.6553434e-03, -1.9514188e-02,  1.1532067e-02, -4.1229022e-03,
                                2.3510400e-02,  1.7623754e-02, -1.4590423e-02,  1.1135901e-03});
    vector<float_complex> general_exp_moments(trig_moments.size());

    trigonometricToExponentialMoments(trig_moments.size(), general_exp_moments.data(), trig_moments.data());

    vector<float_complex> computed_trig_moments(trig_moments.size());
    exponentialToTrigonometricMoments(trig_moments.size(), computed_trig_moments.data(), general_exp_moments.data());

    for (size_t l = 0; l < trig_moments.size(); ++l)
        ASSERT_NEAR(trig_moments[l], computed_trig_moments[l].x, 0.01f);
}

TEST(test_mese, levinson4)
{
    vector<float> trig_moments({0.22008814, 0.f, 0.01400074, 0.00271224});
    vector<float_complex> exp_moments(4);
    trigonometricToExponentialMoments4(exp_moments.data(), trig_moments.data());

    vector<float_complex> solution_fixed(4);
    vector<float_complex> solution_general(4);

    levinsonsAlgorithm(4, solution_general.data(), exp_moments.data(), solution_fixed.data() /* temp */);

    levinsonsAlgorithm4(solution_fixed.data(), exp_moments.data());

    for (int i = 0; i < static_cast<int>(exp_moments.size()); ++i)
    {
        ASSERT_TRUE(cut::abs(solution_fixed[i].x - solution_general[i].x) < 1e-2f);
        ASSERT_TRUE(cut::abs(solution_fixed[i].y - solution_general[i].y) < 1e-2f);
    }
}

TEST(test_mese, autocorrelation4)
{
    // Use any random signal for testing
    vector<float_complex> signal({2.f, 4.f, 8.f, 16.f});
    vector<float_complex> solution_fixed(4);
    vector<float_complex> solution_general(4);

    computeAutocorrelation4(solution_fixed.data(), signal.data());
    computeAutocorrelation(4, solution_general.data(), signal.data());

    for (int i = 0; i < static_cast<int>(signal.size()); ++i)
    {
        ASSERT_FLOAT_EQ(solution_fixed[i].x, solution_general[i].x);
        ASSERT_FLOAT_EQ(solution_fixed[i].y, solution_general[i].y);
    }
}

TEST(test_mese, correlation4)
{
    // Use any random signal for testing
    vector<float_complex> lhs({2.f, 4.f, 8.f, 16.f});
    vector<float_complex> rhs({1.f, 42.f, 0.1f, 0.2f});
    vector<float_complex> solution_fixed(4);
    vector<float_complex> solution_general(4);

    computeCorrelation4(solution_fixed.data(), lhs.data(), rhs.data());
    computeCorrelation(4, solution_general.data(), lhs.data(), rhs.data());

    for (int i = 0; i < static_cast<int>(lhs.size()); ++i)
    {
        ASSERT_FLOAT_EQ(solution_fixed[i].x, solution_general[i].x);
        ASSERT_FLOAT_EQ(solution_fixed[i].y, solution_general[i].y);
    }
}

TEST(test_mese, imaginary_correlation4)
{
    // Use any random signal for testing
    vector<float_complex> lhs({2.f, 4.f, 8.f, 16.f});
    vector<float_complex> rhs({1.f, 42.f, 0.1f, 0.2f});
    vector<float> solution_fixed(4);
    vector<float> solution_general(4);

    computeImaginaryCorrelation4(solution_fixed.data(), lhs.data(), rhs.data());
    computeImaginaryCorrelation(4, solution_general.data(), lhs.data(), rhs.data());

    for (int i = 0; i < static_cast<int>(lhs.size()); ++i)
        ASSERT_FLOAT_EQ(solution_fixed[i], solution_general[i]);
}

TEST(test_mese, prepare)
{
    vector<float> moments({0.22008814, 0.f, 0.01400074, 0.00271224});
    vector<float> pmoments_fixed(4);
    vector<float> pmoments_general(4);

    vector<float_complex> temp0(4);
    vector<float_complex> temp1(4);
    vector<float_complex> temp2(4);

    prepareReflectanceSpectrumLagrange4(pmoments_fixed.data(), moments.data());

    prepareMESELagrange(4, pmoments_general.data(), moments.data(), temp0.data(), temp1.data(), temp2.data());

    for (int i = 0; i < static_cast<int>(moments.size()); ++i)
        ASSERT_TRUE(cut::abs(pmoments_fixed[i] - pmoments_general[i]) < 1e-2f);
}

TEST(test_mese, prepare_complex)
{
    vector<float_complex> moments({float_complex(0.22008814, 0.05f), float_complex(-0.02f, 0.01f), float_complex(0.01400074, 0.04f),
                                   float_complex(0.00271224, 0.1f)});
    vector<float_complex> pmoments_fixed(4);
    vector<float_complex> pmoments_general(4);

    float bias = 1e-2f;

    moments[0] = cut::lerp(moments[0], float_complex(0.5f), bias);
    for (int i = 1; i < 4; ++i)
        moments[i] = cut::lerp(moments[i], float_complex(0.0f), bias);
    prepareReflectanceSpectrumLagrange4(pmoments_fixed.data(), moments.data());

    vector<float_complex> temp0(4);
    vector<float_complex> temp1(4);
    vector<float_complex> temp2(4);

    prepareMESELagrange(4, pmoments_general.data(), moments.data(), temp0.data(), temp1.data(), temp2.data());

    for (int i = 0; i < static_cast<int>(moments.size()); ++i)
    {
        ASSERT_NEAR(pmoments_fixed[i].x, pmoments_general[i].x, 1e-2f);
        ASSERT_NEAR(pmoments_fixed[i].y, pmoments_general[i].y, 1e-2f);
    }
}