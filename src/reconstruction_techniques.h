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
#ifndef MRAY_RECONSTRUCTION_TECHNIQUES_H
#define MRAY_RECONSTRUCTION_TECHNIQUES_H

#include "common.h"
#include "cut/strided_array.h"
#include "mese/MESE_dynamic.h"
#include "raymarch_common.h"
#include "parameters.h"

template <typename T>
struct is_single_scattering_cached
{
    static const bool value = false;
};

/**
 * Truncated Fourier reconstruction.
 */
template<typename float_array>
struct TruncatedFourierReconstruction
{
    int num_moments;
    const float_array trig_moments;
    Vec2f bound;

    FUNC TruncatedFourierReconstruction(int num, const float_array moments, Vec2f b)
        : num_moments(num)
        , trig_moments(moments)
        , bound(b)
    { }

    FUNC float reconstruct(float p) const
    {
        p = cut::clamp(p, get_min_phase(), get_max_phase());

        auto conjCirclePoint = float_complex(cosf(-p), sinf(-p));
        auto s = evaluateFourierSeries(num_moments, conjCirclePoint, trig_moments);
        return s * (bound.y - bound.x) + bound.x;
    }
};

using TruncatedFourierReconstructionSMEM = TruncatedFourierReconstruction<strided_array<float>>;
using TruncatedFourierReconstructionDefault = TruncatedFourierReconstruction<float*>;

/**
 * Bounded MESE reconstruction (given Lagrange multipliers).
 */
template<typename float_array>
struct MESEReconstruction
{
    int num_moments;
    const float_array pmoments;
    Vec2f bound;

    FUNC MESEReconstruction(int num, const float_array m, Vec2f b)
        : num_moments(num)
        , pmoments(m)
        , bound(b)
    { }

    FUNC float reconstruct(float p) const
    {
        p = cut::clamp(p, get_min_phase(), get_max_phase());

        auto s = evaluateMESELagrange(num_moments, p, pmoments);
        assert(!std::isnan(s));
        return s * (bound.y - bound.x) + bound.x;
    }
};

using MESEReconstructionSMEM = MESEReconstruction<strided_array<float>>;
using MESEReconstructionDefault = MESEReconstruction<const float*>;

template<typename MESEReconstruction>
struct ErrorBoundMESEReconstruction
{
    MESEReconstruction mese;
    float error_bound;

    FUNC ErrorBoundMESEReconstruction(MESEReconstruction mese, float error)
        : mese(mese)
        , error_bound(error)
    { }

    FUNC Vec2f reconstruct_bounds(float p) const
    {
        auto rho = mese.reconstruct(p);
        return Vec2f(cut::clamp(rho - error_bound, mese.bound.x, mese.bound.y),
                     cut::clamp(rho + error_bound,  mese.bound.x, mese.bound.y));
    }
};

template<typename MESEReconstruction>
struct MeanStdDevMESEReconstruction
{
    MESEReconstruction mese;
    float std_deviation;

    FUNC MeanStdDevMESEReconstruction(MESEReconstruction mese, float error)
        : mese(mese)
        , std_deviation(error)
    { }

    FUNC Vec2f reconstruct_bounds(float p) const
    {
        auto rho = mese.reconstruct(p);
        return Vec2f(rho, std_deviation);
    }
};

/**
 * Interpolates a density in the given uncertainty bounds.
 * Used for temporal animation of the uncertainty.
 */
template<typename BoundReconstruction>
struct ErrorBoundInterpolator
{
    BoundReconstruction rec;
    float t; // [0, 1]

    FUNC ErrorBoundInterpolator(BoundReconstruction br, float interp_t)
        : rec(br)
        , t(interp_t)
    { }

    FUNC float reconstruct(float p) const
    {
        auto bounds = rec.reconstruct_bounds(p);
        return cut::lerp(bounds.x, bounds.y, t);
    }
};


#endif // MRAY_RECONSTRUCTION_TECHNIQUES_H
