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
#ifndef MRAY_MOMENT_PREDICTION_CODING_H
#define MRAY_MOMENT_PREDICTION_CODING_H

#include "common.h"
#include "coding_params.h"
#include "mese/complex_algebra.h"

#define PREDICTION_CODING_QUANTIZATION_MIN 0.0f
#define PREDICTION_CODING_QUANTIZATION_MAX 1.0f

struct WarpParameters
{
    int warp_type;
    const float *p;

    FUNC float get_param1(int l) const { return p[(l-1)*2]; }
    FUNC float get_param2(int l) const { return p[(l-1)*2+1]; }
};

/**
 * Transforms the given value x at index l using the specified parameters.
 */
FUNC float prediction_coding_warp(int l, float x, WarpParameters params)
{
    float w;
    if (params.warp_type == CODING_WARP_DEFAULT_TRANSFORMED)
        w = (x + params.get_param1(l)) / (params.get_param2(l) * 2.f);
    else
        w = x * 0.5f + 0.5f;

    return cut::clamp(w, PREDICTION_CODING_QUANTIZATION_MIN, PREDICTION_CODING_QUANTIZATION_MAX);
}

/**
 * Inverts the transforms for a given value x at index l using the specified parameters.
 */
FUNC float prediction_coding_inv_warp(int l, float x, WarpParameters params)
{
    if (params.warp_type == CODING_WARP_DEFAULT_TRANSFORMED)
        return (x * params.get_param2(l) * 2.f) - params.get_param1(l);
    else
        return x * 2.f - 1.f;
}

/**
 * Performs the min/max transformation for all moments.
 */
FUNC void transform_quantization_real_warp(int num_moments, float *out, WarpParameters p)
{
    for (int l = 1; l < num_moments; ++l)
        out[l] = prediction_coding_warp(l, out[l], p);
}

/**
 * Transforms the complex residuals to the real line.
 */
template <typename float_complex_array>
FUNC void transform_quantization_real(int num_moments, const float_complex_array code, float c_0, float *out)
{
    out[0] = c_0;

    auto axis = float_complex(-code[0].y, code[0].x);
    auto length_axis = sqrtf(absSqr(axis));
    auto inv_axis = length_axis / axis;

    for (int l = 1; l < num_moments; ++l)
        out[l] = (code[l] * inv_axis).x;
}

/**
 * Inverts both transformations and returns the complex residuals.
 */
template <typename float_complex_array>
FUNC void transform_dequantization_real(int num_moments, const float *in, float_complex_array out,
                                        WarpParameters params)
{
    float c_0 = in[0];

    float zerothMomentPhase = 3.14159265f * c_0 - 1.57079633f;
    out[0] = float_complex(cosf(zerothMomentPhase), sinf(zerothMomentPhase));
    out[0] = 2.0f * 0.0795774715f * out[0];

    auto axis = float_complex(-out[0].y, out[0].x);
    auto length_axis = sqrtf(absSqr(axis));
    auto factor = axis / length_axis;

    for (int l = 1; l < num_moments; ++l)
    {
        auto ql = in[l];
        ql = prediction_coding_inv_warp(l, ql, params);

        out[l] = ql * factor;
    }
}

/**
 * Performs the coding detailed in Sec. 3.6.1.
 */
template <typename float_complex_array>
FUNC void encode(int num_moments, const float_complex_array exp_moments, float_complex_array code,
                 float_complex_array eval_polynom, float_complex_array temp_flipped_solution)
{
    code[0] = exp_moments[0];

    eval_polynom[0] = float_complex(1.0f / (exp_moments[0].x), 0.0f);
    bool invalid = false;
    for (int l = 1; l < num_moments; ++l)
    {
        float_complex u_l(0.0f);
        for (int k = 0; k < l; ++k)
            u_l = u_l + eval_polynom[k] * exp_moments[l - k];

        code[l] = u_l;

        float absSqr_u_l = absSqr(u_l);
        if (invalid || absSqr_u_l >= 1.0f)
        {
            u_l = 0;
            absSqr_u_l = 0;
            invalid = true;
        }

        float factor = 1.0f / (1.0f - absSqr_u_l);

        for (int k = 1; k < l; ++k)
            temp_flipped_solution[k] = conjugate(eval_polynom[l - k]);
        temp_flipped_solution[l] = float_complex(eval_polynom[0].x, 0.0f);

        eval_polynom[0] = float_complex(factor * eval_polynom[0].x, 0.0f);
        for (int k = 1; k < l; ++k)
            eval_polynom[k] = factor * (eval_polynom[k] - temp_flipped_solution[k] * u_l);
        eval_polynom[l] = factor * (-temp_flipped_solution[l].x * u_l);
    }
}

/**
 * Reverts the coding detailed in Sec. 3.6.1.
 */
template <typename float_complex_array>
FUNC void decode(int num_moments, const float_complex_array code, float_complex_array exp_moments,
                 float_complex_array eval_polynom, float_complex_array temp_flipped_solution)
{
    exp_moments[0] = code[0];
    eval_polynom[0] = float_complex(1.0f / (exp_moments[0].x), 0.0f);

    // To avoid numerical issues...
    constexpr float eps = 1e-6f;

    bool invalid = false;
    for (int l = 1; l < num_moments; ++l)
    {
        auto u_l = code[l];
        assert(!std::isnan(u_l.x));

        float absSqr_u_l = absSqr(u_l);
        if (invalid || absSqr_u_l >= (1.0f - eps))
        {
            u_l = 0;
            absSqr_u_l = 0;
            invalid = true;
        }
        assert(!std::isnan(absSqr_u_l));

        float_complex gamma_l(0.0f);
        for (int k = 1; k < l; ++k)
            gamma_l = gamma_l + eval_polynom[k] * exp_moments[l - k];

        exp_moments[l] = (1.0 / eval_polynom[0]) * (u_l - gamma_l);

        float factor = 1.0f / (1.0f - absSqr_u_l);
        assert(!std::isnan(factor));

        for (int k = 1; k < l; ++k)
            temp_flipped_solution[k] = conjugate(eval_polynom[l - k]);
        temp_flipped_solution[l] = float_complex(eval_polynom[0].x, 0.0f);

        eval_polynom[0] = float_complex(factor * eval_polynom[0].x, 0.0f);
        assert(!std::isinf(eval_polynom[0].x));

        for (int k = 1; k < l; ++k)
            eval_polynom[k] = factor * (eval_polynom[k] - temp_flipped_solution[k] * u_l);
        eval_polynom[l] = factor * (-temp_flipped_solution[l].x * u_l);
    }
}

#endif // MRAY_MOMENT_PREDICTION_CODING_H
