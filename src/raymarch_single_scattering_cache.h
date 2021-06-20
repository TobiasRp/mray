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
#ifndef MRAY_RAYMARCH_SINGLE_SCATTERING_CACHE_H
#define MRAY_RAYMARCH_SINGLE_SCATTERING_CACHE_H

#include "raymarch_common.h"
#include "reconstruction_techniques.h"

/**
 * Ray marching to reconstruct a signal using a moment image. Accumulates transmittance to scatter it to a grid.
 * TODO: Some form of DDA or grid traversal would be better.
 */
template <typename ReconstructionTechnique, typename TransferFunction, typename ScatterGrid>
FUNC void reconstruct_single_scattering_ray(cut::Ray ray, Range<Vec3f> domain, float znear, ReconstructionTechnique rec,
                                            const TransferFunction &tf, ScatterGrid &grid,
                                            const ReconstructionParameters &rec_params)
{
    float t_near, t_far;
    auto hit = intersect_AABB(ray, domain.min, domain.max, t_near, t_far);
    t_near = cut::max(t_near, znear);

    auto trange = cut::Vec2f(t_near, t_far);

    float world_step_size = rec_params.step_size;

    if (hit && (t_far > 0))
    {
        float t = t_near;
        float p = to_phase(trange.x, trange.y, t);
        float phase_step = get_phase_step(trange.x, trange.y, world_step_size);

        float tau = 0.0f;
        float fs = rec.reconstruct(p);
        float bs;

        while (t <= t_far)
        {
            t += world_step_size;
            p += phase_step;

            bs = rec.reconstruct(p);

            float sigma_t = tf.sample(fs, bs).w;

            tau += sigma_t;
            fs = bs;

            grid.scatter(ray.origin + t * ray.dir, exp(-tau));
        }
    }
}

#endif // MRAY_RAYMARCH_SINGLE_SCATTERING_CACHE_H
