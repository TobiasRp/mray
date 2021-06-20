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
#ifndef MRAY_TRANSFER_FUNCTION_H
#define MRAY_TRANSFER_FUNCTION_H

#include "common.h"

struct TransferFunction
{
    static constexpr int RESOLUTION = 256;

    array<Vec4f, RESOLUTION * RESOLUTION> data;

    Vec4f get(int x, int y) const
    {
        x = cut::clamp(x, 0, RESOLUTION - 1);
        y = cut::clamp(y, 0, RESOLUTION - 1);
        return data[y * RESOLUTION + x];
    }

    // Samples a 2D table using bilinear interpolation
    Vec4f sample(float fs, float bs) const
    {
        Vec2f ip = Vec2f(fs, bs) * static_cast<float>(RESOLUTION);
        Vec2f a = ip - Vec2f(std::floor(ip.x), std::floor(ip.y));

        int xl = std::floor(ip.x);
        int yl = std::floor(ip.y);

        // Bilinear interpolation
        auto ll = get(xl, yl);
        auto lr = get(xl, yl + 1);
        auto rr = get(xl + 1, yl + 1);
        auto rl = get(xl + 1, yl);

        return (1.f - a.x) * (1.f - a.y) * ll + a.x * (1.f - a.y) * lr + (1.f - a.x) * a.y * rl + a.x * a.y * rr;
    }
};

extern TransferFunction read_from_disk_1D(const char *filepath);

extern TransferFunction read_from_disk(const char *filepath, float step_size);

extern TransferFunction read_from_disk_interpolate(const char *filepath_1, const char *filepath_2, float t, float step_size);

extern TransferFunction read_from_disk_uncertainty(const char *filepath, float step_size, int error_type);

#endif // MRAY_TRANSFER_FUNCTION_H
