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
#ifndef MRAY_RAY_HISTOGRAM_IMAGE_DEVICE_CUH
#define MRAY_RAY_HISTOGRAM_IMAGE_DEVICE_CUH

#include "ray_histogram_image.h"

struct RayHistogramImageDeviceView
{
    int width, height;
    Range<Vec3f> domain;

    const uint32_t *indices;
    const SubFrustum *frusta;
    const uint16_t *frequencies;
    const uint16_t *bin_ids;

    __device__ PixelRayHistogramView get_view(int x, int y) const
    {
        return PixelRayHistogramView(indices, frusta, frequencies, bin_ids, y * width + x);
    }
};

struct RayHistogramImageDevice
{
    RayHistogramImageDeviceView view;

    cut::dev_ptr<uint32_t> d_indices;
    cut::dev_ptr<SubFrustum> d_frusta;
    cut::dev_ptr<uint16_t> d_frequencies;
    cut::dev_ptr<uint16_t> d_bin_ids;


    RayHistogramImageDevice(const RayHistogramImageHost &himg)
        : d_indices(himg.indices.size())
        , d_frusta(himg.frusta.size())
        , d_frequencies(himg.frequencies.size())
        , d_bin_ids(himg.bin_ids.size())
    {
        view.width = himg.width;
        view.height = himg.height;
        view.domain = himg.domain;

        d_indices.loadFromHost(himg.indices.data(), himg.indices.size());
        d_frusta.loadFromHost(himg.frusta.data(), himg.frusta.size());
        d_frequencies.loadFromHost(himg.frequencies.data(), himg.frequencies.size());
        d_bin_ids.loadFromHost(himg.bin_ids.data(), himg.bin_ids.size());

        view.indices = d_indices.get();
        view.frusta = d_frusta.get();
        view.frequencies = d_frequencies.get();
        view.bin_ids = d_bin_ids.get();
    }
};

#endif // MRAY_RAY_HISTOGRAM_IMAGE_DEVICE_CUH
