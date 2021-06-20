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
#ifndef MRAY_COLOR_IMAGE_H
#define MRAY_COLOR_IMAGE_H

#include "common.h"

struct ColorImageHost
{
    struct rgba8_t
    {
        unsigned char r;
        unsigned char g;
        unsigned char b;
        unsigned char a;

        rgba8_t() = default;

        rgba8_t(Vec4f c)
        {
            r = static_cast<unsigned char>(c.x * 255.f);
            g = static_cast<unsigned char>(c.y * 255.f);
            b = static_cast<unsigned char>(c.z * 255.f);
            a = static_cast<unsigned char>(c.w * 255.f);
        }
    };

    int width, height;
    vector<rgba8_t> rgba;

    ColorImageHost(int w, int h)
        : width(w)
        , height(h)
        , rgba(w * h)
    {
    }

    void set_color(int x, int y, Vec4f c)
    {
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        rgba[y * width + x] = rgba8_t(cut::clamp(c, 0.0f, 1.0f));
    }

    rgba8_t get_color(int x, int y) const
    {
        return rgba[y * width + x];
    }
};

extern void write_PPM(string file, const ColorImageHost &img, Vec3f background_color);

#endif // MRAY_COLOR_IMAGE_H
