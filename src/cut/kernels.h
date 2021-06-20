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
#ifndef KERNELS_H
#define KERNELS_H

#include "cuda_common.h"

namespace cut
{

namespace kernels
{

#define EPANECHNIKOW_SUPPORT 1.0f
#define CUBIC_SPLINE_SUPPORT 2.0f
#define QUINTIC_SPLINE_SUPPORT 3.0f

/**
 * @brief Evaluates the Epanechnikov kernel
 * @note The support of the kernel is [0,1)
 * @return
 */
FUNC float epanechnikov(float x)
{
    return 0.75f * (1 - x * x);
}

/**
 * @brief Evaluates a cubic spline (commonly used in SPH simulations)
 *
 * @note The support of the spline is [0,2]
 */
FUNC float cubicSpline(float x)
{
    constexpr float sigma = 0.5641895835477563f; //(1/sqrt(pi))
    if (x < 1.f)
        return sigma * (1.f - (3.f / 2.f) * x * x * (1.f - x * 0.5f));
    else if (x < 2.f)
        return sigma / 4.f * (2.f - x) * (2.f - x) * (2.f - x);
    else
        return 0.0f;
}

/**
 * @brief Evaluates a quintic spline (commonly used in SPH simulations).
 *
 * @note The support of the spline is [0,3]
 */
FUNC float quinticSpline(float q)
{
    constexpr float sigma = 1.0f / 120.f;

    float q3 = (3. - q);
    float q2 = (2. - q);
    float q1 = (1. - q);

    if ((q >= 0.0) && (q < 1.0))
    {
        return sigma * (q3 * q3 * q3 * q3 * q3 - 6. * q2 * q2 * q2 * q2 * q2 + 15. * q1 * q1 * q1 * q1 * q1);
    }
    else if ((q >= 1.0) && (q < 2.0))
    {
        return sigma * (q3 * q3 * q3 * q3 * q3 - 6. * q2 * q2 * q2 * q2 * q2);
    }
    else if ((q >= 2.0) && (q < 3.0))
    {
        return sigma * (q3 * q3 * q3 * q3 * q3);
    }
    return 0.0;
}

} // namespace kernels
} // namespace cut

#endif // KERNELS_H
