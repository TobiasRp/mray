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
#ifndef MRAY_MOMENT_PREPARATION_H
#define MRAY_MOMENT_PREPARATION_H

#include "raymarch_common.h"
#include "moment_prediction_coding.h"
#include "mese/MESE_dynamic.h"

template <typename float_complex_array>
FUNC void prepare_moments_from_pred_coding(int num_moments, const float *moments, float *pmoments,
                                           float_complex_array temp_code, float_complex_array temp_exp_moments,
                                           float_complex_array temp_eval_poly, float_complex_array temp,
                                           WarpParameters params)
{
    transform_dequantization_real(num_moments, moments, temp_code, params);

    decode(num_moments, temp_code, temp_exp_moments, temp_eval_poly, temp);

    for (int i = 0; i < num_moments; ++i)
        temp_eval_poly[i] = 6.28318531f * temp_eval_poly[i];

    computeAutocorrelation(num_moments, temp, temp_eval_poly);
    temp_exp_moments[0] = 0.5f * temp_exp_moments[0];
    computeImaginaryCorrelation(num_moments, pmoments, temp, temp_exp_moments);
    float normalizationFactor = 1.0f / (3.14159265f * temp_eval_poly[0].x);
    for (int i = 0; i < num_moments; ++i)
        pmoments[i] = normalizationFactor * pmoments[i];

    assert(!std::isnan(pmoments[0]));
}

template <typename float_complex_array>
FUNC void prepare_moments_from_std_coding(int num_moments, float *moments, float *pmoments, float_complex_array temp0,
                         float_complex_array temp1, float_complex_array temp2, float bias)
{
    moments[0] = cut::lerp(moments[0], 0.5f, bias);
    for (int i = 1; i < num_moments; ++i)
        moments[i] = cut::lerp(moments[i], 0.0f, bias);

    prepareMESELagrange(num_moments, pmoments, moments, temp0, temp1, temp2);
}

#endif // MRAY_MOMENT_PREPARATION_H
