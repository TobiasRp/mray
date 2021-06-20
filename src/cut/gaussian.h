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
#ifndef MRAY_GAUSSIAN_H
#define MRAY_GAUSSIAN_H

#include "cuda_common.h"

namespace cut
{

template <typename T> FUNC T normal_cdf(T x)
{
    // constants
    static constexpr T a1 = 0.254829592;
    static constexpr T a2 = -0.284496736;
    static constexpr T a3 = 1.421413741;
    static constexpr T a4 = -1.453152027;
    static constexpr T a5 = 1.061405429;
    static constexpr T p = 0.3275911;

    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = std::abs(x) / 1.41421356237; // std::sqrt(static_cast<T>(2.0));

    // A&S formula 7.1.26
    T t = 1.0 / (1.0 + p * x);
    T y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * std::exp(-x * x);

    return 0.5 * (1.0 + sign * y);
}

template <typename T> FUNC T normal_cdf(T x, T mean, T sigma)
{
    return normal_cdf((x - mean) / sigma);
}

#include <cmath>

// https://stackoverflow.com/questions/27229371/inverse-error-function-in-c

///* compute inverse error functions with maximum error of 2.35793 ulp */
FUNC float myerfinvf (float a)
{
    float p, r, t;
    t = fmaf (a, 0.0f - a, 1.0f);
    t = logf (t);
    if (fabsf(t) > 6.125f) { // maximum ulp error = 2.35793
        p =              3.03697567e-10f; //  0x1.4deb44p-32
        p = fmaf (p, t,  2.93243101e-8f); //  0x1.f7c9aep-26
        p = fmaf (p, t,  1.22150334e-6f); //  0x1.47e512p-20
        p = fmaf (p, t,  2.84108955e-5f); //  0x1.dca7dep-16
        p = fmaf (p, t,  3.93552968e-4f); //  0x1.9cab92p-12
        p = fmaf (p, t,  3.02698812e-3f); //  0x1.8cc0dep-9
        p = fmaf (p, t,  4.83185798e-3f); //  0x1.3ca920p-8
        p = fmaf (p, t, -2.64646143e-1f); // -0x1.0eff66p-2
        p = fmaf (p, t,  8.40016484e-1f); //  0x1.ae16a4p-1
    } else { // maximum ulp error = 2.35456
        p =              5.43877832e-9f;  //  0x1.75c000p-28
        p = fmaf (p, t,  1.43286059e-7f); //  0x1.33b458p-23
        p = fmaf (p, t,  1.22775396e-6f); //  0x1.49929cp-20
        p = fmaf (p, t,  1.12962631e-7f); //  0x1.e52bbap-24
        p = fmaf (p, t, -5.61531961e-5f); // -0x1.d70c12p-15
        p = fmaf (p, t, -1.47697705e-4f); // -0x1.35be9ap-13
        p = fmaf (p, t,  2.31468701e-3f); //  0x1.2f6402p-9
        p = fmaf (p, t,  1.15392562e-2f); //  0x1.7a1e4cp-7
        p = fmaf (p, t, -2.32015476e-1f); // -0x1.db2aeep-3
        p = fmaf (p, t,  8.86226892e-1f); //  0x1.c5bf88p-1
    }
    r = a * p;
    return r;
}


} // namespace cut

#endif // MRAY_GAUSSIAN_H
