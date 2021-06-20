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
#ifndef MRAY_SINGLE_SCATTERING_IMAGE_H
#define MRAY_SINGLE_SCATTERING_IMAGE_H

#include "cut/camera.h"
#include "cut/raytracing.h"

#include "reconstruction_techniques.h"
#include "moment_image.h"

struct NoSingleScattering
{
    NoSingleScattering get_view() { return NoSingleScattering{}; }

    constexpr FUNC bool enabled() const { return false; }
    constexpr FUNC bool use_cache() const { return false; }

    template <typename float_array> FUNC void set_smem(float_array mem) { UNUSED(mem); }

    void prepare(float bias) { UNUSED(bias); }

    FUNC int get_max_moments() const { return 0; }
};

struct SingleScatteringImageHost
{
    cut::Camera cam;
    MomentImageHost img;

    SingleScatteringImageHost(cut::Camera cam, const MomentImageHost &img)
        : cam(cam)
        , img(img)
    {
    }

    constexpr bool enabled() const { return true; }
    constexpr bool use_cache() const { return false; }

    Vec2f get_ndc(Vec3f pt) const
    {
        auto proj = cam.transform * Vec4f(pt, 1.0f);
        return Vec2f(proj.x, proj.y);
    }

    cut::Ray get_ray(Vec2f ndc) const
    {
        cut::Ray ray;
        ray.origin = cam.position + ndc.x * cam.right + ndc.y * cam.up;
        ray.dir = cam.dir;
        return ray;
    }

    cut::Vec2i get_nearest_pixel(Vec2f ndc) const
    {
        return cut::Vec2i(cut::clamp(static_cast<int>((ndc.x * 0.5f + 0.5f) * (img.width - 1)), 0, img.width - 1),
                          cut::clamp(static_cast<int>((ndc.y * 0.5f + 0.5f) * (img.height - 1)), 0, img.height - 1));
    }

    const float *get_moments(cut::Vec2i p) const { return &img.data[img.get_idx(p.x, p.y)]; }

    int get_num_moments(cut::Vec2i p) const { return img.get_num_moments(p.x, p.y); }

    Vec2f get_bound(int x, int y) const { return img.get_bounds(x, y); }

    MESEReconstructionDefault get_reconstruction(Vec2f ndc) const
    {
        auto p = get_nearest_pixel(ndc);
        return MESEReconstructionDefault(get_num_moments(p), &img.data[img.get_idx(p.x, p.y)], get_bound(p.x, p.y));
    }

    void prepare(float bias);
};

#endif // MRAY_SINGLE_SCATTERING_IMAGE_H
