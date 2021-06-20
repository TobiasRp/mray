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
#ifndef MRAY_COLOR_IMAGE_DEVICE_CUH
#define MRAY_COLOR_IMAGE_DEVICE_CUH

#include "color_image.h"

struct ColorImageDeviceView
{
    int width, height;
    uchar4 *rgba;

    __device__ void set_color(int x, int y, Vec4f c)
    {
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        c = cut::clamp(c, 0.0f, 1.0f);
        rgba[y * width + x] =
            make_uchar4(static_cast<unsigned char>(c.x * 255.f), static_cast<unsigned char>(c.y * 255.f),
                        static_cast<unsigned char>(c.z * 255.f), static_cast<unsigned char>(c.w * 255.f));
    }
};

struct ColorImageDevice
{
    ColorImageDeviceView view;

    cut::dev_ptr<uchar4> mem;

    ColorImageDevice(const ColorImageHost &img)
    : mem(img.width * img.height)
    {
        view.width = img.width;
        view.height = img.height;

        mem.loadFromHost(reinterpret_cast<const uchar4*>(img.rgba.data()), img.width * img.height);
        view.rgba = mem.get();
    }

    void copy_back(ColorImageHost &img) const
    {
        mem.copyToHost(reinterpret_cast<uchar4*>(img.rgba.data()), img.width * img.height);
    }
};

#endif // MRAY_COLOR_IMAGE_DEVICE_CUH
