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

/**
 * Implementation of bounded MESE based on code from Christoph Peters.
 */
#ifndef MRAY_MESE_DYNAMIC_H
#define MRAY_MESE_DYNAMIC_H

#include <cmath>
#include "complex_algebra.h"

template <typename float_complex_array>
FUNC void trigonometricToExponentialMoments(int num_moments, float_complex_array pOutExponentialMoment,
                                            const float *pTrigonometricMoment)
{
    float zerothMomentPhase = 3.14159265f * pTrigonometricMoment[0] - 1.57079633f;
    pOutExponentialMoment[0] = float_complex(cosf(zerothMomentPhase), sinf(zerothMomentPhase));
    pOutExponentialMoment[0] = 0.0795774715f * pOutExponentialMoment[0];

    for (int l = 1; l < num_moments; ++l)
    {
        pOutExponentialMoment[l] = 0.0f;
        for (int j = 0; j < l; ++j)
            pOutExponentialMoment[l] =
                pOutExponentialMoment[l] + (l - j) * pOutExponentialMoment[j] * pTrigonometricMoment[l - j];

        pOutExponentialMoment[l] = pOutExponentialMoment[l] * float_complex(0.0f, 2.0f * M_PI / static_cast<float>(l));
    }

    pOutExponentialMoment[0] = 2.0f * pOutExponentialMoment[0];
}

template <typename float_complex_array>
FUNC void trigonometricToExponentialMoments(int num_moments, float_complex_array pOutExponentialMoment,
                                            const float_complex_array pTrigonometricMoment)
{
    float zerothMomentPhase = 3.14159265f * pTrigonometricMoment[0].x - 1.57079633f;
    pOutExponentialMoment[0] = float_complex(cosf(zerothMomentPhase), sinf(zerothMomentPhase));
    pOutExponentialMoment[0] = 0.0795774715f * pOutExponentialMoment[0];

    for (int l = 1; l < num_moments; ++l)
    {
        pOutExponentialMoment[l] = 0.0f;
        for (int j = 0; j < l; ++j)
            pOutExponentialMoment[l] =
                pOutExponentialMoment[l] + (l - j) * pOutExponentialMoment[j] * pTrigonometricMoment[l - j];

        pOutExponentialMoment[l] = pOutExponentialMoment[l] * float_complex(0.0f, 2.0f * M_PI / static_cast<float>(l));
    }

    pOutExponentialMoment[0] = 2.0f * pOutExponentialMoment[0];
}

template <typename float_complex_array>
FUNC void exponentialToTrigonometricMoments(int num_moments, float_complex_array outTrigmoments,
                                            const float_complex_array exp_moments)
{
    auto gamma_0 = exp_moments[0] / 2.f;

    outTrigmoments[0] = gamma_0 / 0.0795774715f;

    float zerothMomentPhase = atan2(outTrigmoments[0].y, outTrigmoments[0].x);
    outTrigmoments[0] = (zerothMomentPhase + 1.57079633f) / 3.14159265f;
    outTrigmoments[0].y = 0.0f;

    assert(!std::isnan(zerothMomentPhase));

    for (int l = 1; l < num_moments; ++l)
    {
        outTrigmoments[l] = 0.0f;
        for (int j = 1; j < l; ++j)
            outTrigmoments[l] = outTrigmoments[l] + (l - j) / (l * gamma_0) * exp_moments[j] * outTrigmoments[l - j];
        outTrigmoments[l] = exp_moments[l] / (float_complex(0.0f, 2.0f * M_PI) * gamma_0) - outTrigmoments[l];
    }
}

template <typename float_complex_array>
FUNC void levinsonsAlgorithm(int num_moments, float_complex_array pOutSolution, const float_complex_array pFirstColumn,
                             float_complex_array temp_flipped_solution)
{
    pOutSolution[0] = float_complex(1.0f / (pFirstColumn[0].x), 0.0f);

    bool invalid = false;
    for (int j = 1; j < num_moments; ++j)
    {
        float_complex u_l(0.0f);
        if (!invalid)
        {
            for (int k = 0; k < j; ++k)
                u_l = u_l + pOutSolution[k] * pFirstColumn[j - k];
        }

        float absSqr_u_l = absSqr(u_l);
        if (absSqr_u_l >= 1.0f)
        {
            u_l = 0;
            absSqr_u_l = 0;
            invalid = true;
        }

        float factor = 1.0f / (1.0f - absSqr_u_l);

        for (int k = 1; k < j; ++k)
            temp_flipped_solution[k] = conjugate(pOutSolution[j - k]);
        temp_flipped_solution[j] = float_complex(pOutSolution[0].x, 0.0f);

        pOutSolution[0] = float_complex(factor * pOutSolution[0].x, 0.0f);
        for (int k = 1; k < j; ++k)
            pOutSolution[k] = factor * (pOutSolution[k] - temp_flipped_solution[k] * u_l);
        pOutSolution[j] = factor * (-temp_flipped_solution[j].x * u_l);
    }
}

template <typename float_complex_array>
FUNC void computeAutocorrelation(int num_moments, float_complex_array pOutAutocorrelation,
                                 const float_complex_array pSignal)
{
    for (int k = 0; k < num_moments; ++k)
    {
        pOutAutocorrelation[k] = 0.0f;
        for (int j = 0; j < num_moments - k; ++j)
            pOutAutocorrelation[k] = pOutAutocorrelation[k] + pSignal[j] * conjugate(pSignal[j + k]);
    }
}

template <typename float_complex_array>
FUNC void computeCorrelation(int num_moments, float_complex_array pOutCorrelation, const float_complex_array pLHS,
                             const float_complex_array pRHS)
{
    for (int k = 0; k < num_moments; ++k)
    {
        pOutCorrelation[k] = 0.0f;
        for (int j = 0; j < num_moments - k; ++j)
            pOutCorrelation[k] = pOutCorrelation[k] + pLHS[k + j] * pRHS[j];
    }
}

template <typename float_complex_array>
FUNC void computeImaginaryCorrelation(int num_moments, float *pOutCorrelation, const float_complex_array pLHS,
                                      const float_complex_array pRHS)
{
    for (int k = 0; k < num_moments; ++k)
    {
        pOutCorrelation[k] = 0.0f;
        for (int j = 0; j < num_moments - k; ++j)
            pOutCorrelation[k] += pLHS[k + j].x * pRHS[j].y + pLHS[k + j].y * pRHS[j].x;
    }
}

template <typename float_complex_array>
FUNC void prepareMESELagrange(int num_moments, float_complex_array pOutLagrangeMultiplier,
                              const float_complex_array pTrigonometricMoment, float_complex_array pExponentialMoment,
                              float_complex_array pEvaluationPolynomial, float_complex_array pAutocorrelation)
{
    trigonometricToExponentialMoments(num_moments, pExponentialMoment, pTrigonometricMoment);
    levinsonsAlgorithm(num_moments, pEvaluationPolynomial, pExponentialMoment, pAutocorrelation /*temp memory!*/);
    for (int i = 0; i < num_moments; ++i)
        pEvaluationPolynomial[i] = 6.28318531f * pEvaluationPolynomial[i];

    computeAutocorrelation(num_moments, pAutocorrelation, pEvaluationPolynomial);
    pExponentialMoment[0] = 0.5f * pExponentialMoment[0];
    computeCorrelation(num_moments, pOutLagrangeMultiplier, pAutocorrelation, pExponentialMoment);
    float normalizationFactor = 1.0f / (3.14159265f * pEvaluationPolynomial[0].x);

    for (int i = 0; i < num_moments; ++i)
        pOutLagrangeMultiplier[i] = normalizationFactor * (float_complex(-0.0f, -1.0f) * pOutLagrangeMultiplier[i]);

    pOutLagrangeMultiplier[0] = float_complex(pOutLagrangeMultiplier[0].x, 0.0f);
}

template <typename float_complex_array>
FUNC void prepareMESELagrange(int num_moments, float *pOutLagrangeMultiplier, const float *pTrigonometricMoment,
                              float_complex_array pExponentialMoment, float_complex_array pEvaluationPolynomial,
                              float_complex_array pAutocorrelation)
{
    trigonometricToExponentialMoments(num_moments, pExponentialMoment, pTrigonometricMoment);

    levinsonsAlgorithm(num_moments, pEvaluationPolynomial, pExponentialMoment, pAutocorrelation /*temp memory!*/);
    for (int i = 0; i < num_moments; ++i)
        pEvaluationPolynomial[i] = 6.28318531f * pEvaluationPolynomial[i];

    computeAutocorrelation(num_moments, pAutocorrelation, pEvaluationPolynomial);
    pExponentialMoment[0] = 0.5f * pExponentialMoment[0];
    computeImaginaryCorrelation(num_moments, pOutLagrangeMultiplier, pAutocorrelation, pExponentialMoment);
    float normalizationFactor = 1.0f / (3.14159265f * pEvaluationPolynomial[0].x);
    for (int i = 0; i < num_moments; ++i)
        pOutLagrangeMultiplier[i] = normalizationFactor * pOutLagrangeMultiplier[i];
}

template <typename float_array>
FUNC float evaluateFourierSeries(int num_moments, const float_complex circlePoint,
                                 const float_array pFourierCoefficient)
{
    float result = 0;
    auto circlePointPower = float_complex(1.f);
    for (int l = 1; l < num_moments; ++l)
    {
        circlePointPower = circlePointPower * circlePoint;
        result += pFourierCoefficient[l] * circlePointPower.x;
    }

    return 2.0f * result + pFourierCoefficient[0];
}

template <typename float_complex_array>
FUNC float evaluateFourierSeriesComplex(int num_moments, const float_complex circlePoint,
                                        const float_complex_array pFourierCoefficient)
{
    float result = 0;
    auto circlePointPower = float_complex(1.f);
    for (int l = 1; l < num_moments; ++l)
    {
        circlePointPower = circlePointPower * circlePoint;
        result += pFourierCoefficient[l].x * circlePointPower.x - pFourierCoefficient[l].y * circlePointPower.y;
    }
    return 2.0f * result + pFourierCoefficient[0].x;
}

template <typename float_array>
FUNC float evaluateMESELagrange(int num_moments, const float phase, const float_array pLagrangeMultiplier)
{
    float_complex conjCirclePoint;
    conjCirclePoint = float_complex(cosf(-phase), sinf(-phase));
    float lagrangeSeries;
    lagrangeSeries = evaluateFourierSeries(num_moments, conjCirclePoint, pLagrangeMultiplier);
    return fast_atan(lagrangeSeries) * 0.318309886f + 0.5f;
}

template <typename float_complex_array>
FUNC float evaluateMESELagrangeComplex(int num_moments, const float phase,
                                       const float_complex_array pLagrangeMultiplier)
{
    float_complex conjCirclePoint;
    conjCirclePoint = float_complex(cosf(-phase), sinf(-phase));
    float lagrangeSeries;
    lagrangeSeries = evaluateFourierSeriesComplex(num_moments, conjCirclePoint, pLagrangeMultiplier);
    return fast_atan(lagrangeSeries) * 0.318309886f + 0.5f;
}

#endif // MRAY_MESE_DYNAMIC_H
