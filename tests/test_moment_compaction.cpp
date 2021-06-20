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
#include "moment_compaction.h"

TEST(test_moment_compaction, test_exp_to_bounded)
{
    vector<float> trig_moments({1.9176584e-01, -3.7339486e-02, -1.4143054e-02, -1.6614912e-02, -2.2262430e-02,
                                -1.2005451e-02, -7.5180288e-03, 2.5505440e-02, 2.8378680e-02, 1.6212923e-02});

    vector<float_complex> exp_moments(trig_moments.size());
    trigonometricToExponentialMoments(trig_moments.size(), exp_moments.data(), trig_moments.data());

    vector<float_complex> trig_moments2(trig_moments.size());
    trig_moments2[0] = trig_moments[0];

    exponential_to_bounded_moments(trig_moments.size(), exp_moments.data(), trig_moments2.data(), 1);
}

TEST(test_moment_compaction, test_rmse)
{
    vector<float> trig_moments({1.9176584e-01, -3.7339486e-02, -1.4143054e-02, -1.6614912e-02, -2.2262430e-02,
                                -1.2005451e-02, -7.5180288e-03, 2.5505440e-02, 2.8378680e-02, 1.6212923e-02});

    vector<float_complex> exp_moments(trig_moments.size());
    trigonometricToExponentialMoments(trig_moments.size(), exp_moments.data(), trig_moments.data());

    vector<float_complex> levinson(trig_moments.size());

    vector<float_complex> extended_moments2(trig_moments.size());
    evaluate_unknown_moments(2, exp_moments.data(), 10, extended_moments2.data(), levinson.data());

    vector<float_complex> extended_trig_moments2(trig_moments.size());
    for (int l = 0; l < 2; ++l)
        extended_trig_moments2[l] = trig_moments[l];

    exponential_to_bounded_moments(trig_moments.size(), extended_moments2.data(), extended_trig_moments2.data(), 2);

    vector<float_complex> extended_moments8(trig_moments.size());
    evaluate_unknown_moments(8, exp_moments.data(), 10, extended_moments8.data(), levinson.data());

    vector<float_complex> extended_trig_moments8(trig_moments.size());
    for (int l = 0; l < 8; ++l)
        extended_trig_moments8[l] = trig_moments[l];

    exponential_to_bounded_moments(trig_moments.size(), extended_moments8.data(), extended_trig_moments8.data(), 8);

    float rmse2 = rmse(2, trig_moments.size(), extended_trig_moments2.data(), trig_moments.data());
    float rmse8 = rmse(8, trig_moments.size(), extended_trig_moments8.data(), trig_moments.data());

    ASSERT_LT(rmse8, rmse2);

    int num8 = determine_best_number_moments(trig_moments.size(), trig_moments.data(), rmse8, exp_moments.data(),
                                            extended_moments8.data(), levinson.data());
    UNUSED(num8);
//    ASSERT_EQ(num8, 8);
}

TEST(test_moment_compaction, test_compaction_bias)
{
    vector<float> trig_moments({0.1983167 ,  0.03171505, -0.12985915, -0.05961903,  0.00244115,
                                0.03245064,  0.05718907,  0.00187155, -0.03087601, -0.01589895,
                                -0.00614345,  0.03241706,  0.00969719, -0.05645619, -0.00104215,
                                0.05102183,  0.0024777 , -0.00473271, -0.00740673, -0.03612704,
                                0.00892478,  0.03186798, -0.00999953, -0.00158817,  0.00772654,
                                -0.01333004,  0.00057419,  0.00336324, -0.00309478,  0.01228757,
                                -0.0097815 , -0.01696295,  0.02472721,  0.00776743, -0.02235641,
                                0.00757752,  0.0051902 , -0.01507921,  0.00782701,  0.00666146,
                                -0.00656872,  0.00910221, -0.00225353, -0.01672122,  0.00568609,
                                0.01201205,  0.00335739, -0.00337378, -0.01899173, -0.00042338,
                                0.02588879, -0.00246766, -0.01555048,  0.00718114, -0.00284976,
                                -0.00644474,  0.01315433, -0.0008673 , -0.00954569,  0.00664819,
                                -0.0009906 , -0.00470641,  0.00752277,  0.00026853, -0.00366214,
                                -0.00093034, -0.00607619,  0.0047634 ,  0.00832724, -0.00576877,
                                0.0029087 ,  0.00375556, -0.01557173, -0.00113307,  0.01518101,
                                -0.00198062, -0.00516025,  0.00379387, -0.00100452, -0.00159853,
                                0.00050362, -0.00037362,  0.00031693, -0.00424262, -0.00052471,
                                0.01026814,  0.00262053, -0.0067416 , -0.00321998, -0.003694  ,
                                -0.0007162 ,  0.00785196,  0.00396622, -0.00382737, -0.00236847,
                                0.00189891,  0.00054114, -0.00465627, -0.00143932,  0.00462011});

    vector<float_complex> exp_moments(trig_moments.size());
    trigonometricToExponentialMoments(trig_moments.size(), exp_moments.data(), trig_moments.data());

    vector<float_complex> levinson(trig_moments.size());

    vector<float_complex> extended_moments(trig_moments.size());
    evaluate_unknown_moments(12, exp_moments.data(), trig_moments.size(), extended_moments.data(), levinson.data());

    vector<float_complex> extended_trig_moments(trig_moments.size());
    for (int l = 0; l < 12; ++l)
        extended_trig_moments[l] = trig_moments[l];

    exponential_to_bounded_moments(trig_moments.size(), extended_moments.data(), extended_trig_moments.data(), 8);

    float error = rmse(12, trig_moments.size(), extended_trig_moments.data(), trig_moments.data());

    ASSERT_LT(error, 1.0f);
}
