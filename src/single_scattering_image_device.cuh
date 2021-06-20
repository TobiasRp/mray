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
#ifndef MRAY_SINGLE_SCATTERING_IMAGE_DEVICE_CUH
#define MRAY_SINGLE_SCATTERING_IMAGE_DEVICE_CUH

#include "single_scattering_image.h"
#include "moment_image_device.cuh"
#include "cut/strided_array.h"

struct SingleScatteringImageDeviceView
{
    cut::Camera cam;
    MomentImageDeviceView img;

    mutable strided_array<float> smem;

    constexpr FUNC bool enabled() const { return true; }
    constexpr FUNC bool use_cache() const { return false; }

    __device__ void set_smem(strided_array<float> mem) { smem = mem; }

    __device__ Vec2f get_ndc(Vec3f pt) const
    {
        auto proj = cam.transform * Vec4f(pt, 1.0f);
        return Vec2f(proj.x / proj.w, proj.y / proj.w);
    }

    __device__ cut::Ray get_ray(Vec2f ndc) const
    {
        cut::Ray ray;
        ray.origin = cam.position + ndc.x * cam.right + ndc.y * cam.up;
        ray.dir = cam.dir;
        return ray;
    }

    __device__ cut::Vec2i get_nearest_pixel(Vec2f ndc) const
    {
        return cut::Vec2i(cut::clamp(static_cast<int>((ndc.x * 0.5f + 0.5f) * (img.width - 1)), 0, img.width - 1),
                          cut::clamp(static_cast<int>((ndc.y * 0.5f + 0.5f) * (img.height - 1)), 0, img.height - 1));
    }

    __device__ const strided_array<float> get_moments(cut::Vec2i p) const
    {
        // Copy to shared memory
        auto idx = img.get_idx(p.x, p.y);
        for (int m = 0; m < img.get_num_moments(p.x, p.y); ++m)
            smem[m] = img.data[idx + m];
        return smem;
    }

    __device__ int get_num_moments(cut::Vec2i p) const
    {
        return img.get_num_moments(p.x, p.y);
    }

    __device__ Vec2f get_bound(int x, int y) const
    {
        return img.get_bounds(x, y);
    }

    FUNC int get_max_moments() const
    {
        return img.num_moments;
    }

    __device__ MESEReconstructionSMEM get_reconstruction(Vec2f ndc) const
    {
        auto p = get_nearest_pixel(ndc);
        return MESEReconstructionSMEM(img.get_num_moments(p.x, p.y), get_moments(p), get_bound(p.x, p.y));
    }
};

struct SingleScatteringImageDevice
{
    SingleScatteringImageDeviceView view;
    MomentImageDevice d_img;

    SingleScatteringImageDevice(const SingleScatteringImageHost &ss)
        : d_img(ss.img)
    {
        d_img.load_from(ss.img);
        view.img = d_img.view;
        view.cam = ss.cam;
    }

    SingleScatteringImageDeviceView get_view() { return view; }

    void prepare(float bias) {
        d_img.prepare_moments_device(bias);
        view.img = d_img.view;
    }
};

#endif // MRAY_SINGLE_SCATTERING_IMAGE_DEVICE_CUH
