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
#ifndef RAYTRACING_H
#define RAYTRACING_H

#include "camera.h"

namespace cut
{

struct Ray
{
    Vec3f origin;
    Vec3f dir;
};

FUNC Ray get_eye_ray_persp(const Camera &camera, Vec2f ndc)
{
    Ray ray;
    ray.origin = camera.position;

    float x = ndc.x * camera.aspect * camera.scale;
    float y = ndc.y * camera.scale;

    ray.dir = normalize(make_vec3f(camera.transform * Vec4f(x, y, -1.0f, 0.0f)));
    return ray;
}

FUNC Ray get_eye_ray_ortho(const Camera &camera, Vec2f ndc)
{
    Ray ray;
    ray.origin = camera.position + ndc.x * camera.right + ndc.y * camera.up;
    ray.dir = camera.dir;
    return ray;
}

FUNC Ray get_eye_ray(const Camera &camera, Vec2f ndc)
{
    if (camera.orthographic)
        return get_eye_ray_ortho(camera, ndc);
    else
        return get_eye_ray_persp(camera, ndc);
}

FUNC bool intersect_AABB(Ray r, Vec3f bb_min, Vec3f bb_max, float &t_near, float &t_far)
{
    // compute intersection of ray with all six bbox planes
    Vec3f invR = div(Vec3f{1.0f}, r.dir);
    Vec3f tbot = mul(invR, (bb_min - r.origin));
    Vec3f ttop = mul(invR, (bb_max - r.origin));

    // re-order intersections to find smallest and largest on each axis
    Vec3f tmin = min(ttop, tbot);
    Vec3f tmax = max(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
    float smallest_tmax = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));

    t_near = largest_tmin;
    t_far = smallest_tmax;

    return (smallest_tmax > largest_tmin);
}

} // namespace cut

#endif // RAYTRACING_H
