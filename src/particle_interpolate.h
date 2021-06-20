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
#ifndef MRAY_PARTICLE_INTERPOLATE_H
#define MRAY_PARTICLE_INTERPOLATE_H

#include "cut/kernels.h"

FUNC float default_kernel(float x)
{
    return cut::kernels::cubicSpline(x);
}

template <bool USE_VALUES> struct InterpolateParticlesOp
{
    float value = 0.0f;
    float weightSum = 0.0f;
    float inv_h;

    cut::Vec3f p;

    const float *values;

    FUNC void operator()(int nOff, cut::Vec3f pn)
    {
        auto weight = default_kernel(cut::length(pn - p) * inv_h);

        weightSum += weight;
        value += weight * (USE_VALUES ? values[nOff] : 1.f);
    }

    FUNC void normalize()
    {
        if (weightSum != 0.0f)
            value /= weightSum;
    }
};

#endif // MRAY_PARTICLE_INTERPOLATE_H
