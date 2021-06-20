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
#ifndef MRAY_RAYMARCH_GENERATE_SAMPLES_H
#define MRAY_RAYMARCH_GENERATE_SAMPLES_H

#include "cut/raytracing.h"
#include "parameters.h"

/**
 * Ray march and write all samples to memory. For evaluation/debugging.
 */
template <typename Volume>
FUNC void generate_samples_ray(const Volume &volume, cut::Ray ray, float znear, SamplesWriter samples,
                               const GenerationParameters &params)
{
    float t_near, t_far;
    auto hit = intersect_AABB(ray, volume.domain.min, volume.domain.max, t_near, t_far);
    t_near = cut::max(t_near, znear);

    float world_step_size = params.step_size_write_samples;

    int i = 0;
    if (hit && (t_far > 0))
    {
        float t = t_near;

        float fs = map_density(volume.sample(ray.origin + t * ray.dir), params);
        float bs;

        while (t < t_far && i < samples.get_max())
        {
            t += world_step_size;

            bs = map_density(volume.sample(ray.origin + t * ray.dir), params);

            samples.write(i, fs);
            fs = bs;
            ++i;
        }

        // Add last sample exactly at t_far
        if (i < samples.get_max())
        {
            samples.write(i, map_density(volume.sample(ray.origin + t_far * ray.dir), params));
            ++i;
        }
    }

    if (i < samples.get_max())
        samples.write_invalid(i);
}

#endif // MRAY_RAYMARCH_GENERATE_SAMPLES_H
