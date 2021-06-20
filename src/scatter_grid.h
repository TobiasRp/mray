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
#ifndef MRAY_SCATTER_GRID_H
#define MRAY_SCATTER_GRID_H

#include "common.h"
#include "volume.h"

struct ScatterGridHost
{
    Vec3i size;
    Range<Vec3f> domain;
    vector<float> data;

    ScatterGridHost(Vec3i psize, Range<Vec3f> domain, float empty_value)
        : size(psize)
        , domain(domain)
        , data(size.x * size.y * size.z, empty_value)
    {
    }

    RegularGrid to_volume() const
    {
        return RegularGrid(size, domain, data, RegularGrid::DataType::F32);
    }

    Vec3i get_voxel(Vec3f p) const
    {
        Vec3f np = cut::div(p - domain.min, domain.max - domain.min);
        return cut::cast<float, int>(cut::mul(np, cut::cast<int, float>(size)));
    }

    void scatter(Vec3f pt, float T)
    {
        Vec3f np = cut::div(pt - domain.min, domain.max - domain.min);
        Vec3i ip = cut::cast<float, int>(cut::mul(np, cut::cast<int, float>(size)));

        if (ip.x >= 0 && ip.x < size.x && ip.y >= 0 && ip.y < size.y && ip.z >= 0 && ip.z < size.z)
        {
            auto idx = ip.z * size.y * size.x + ip.y * size.x + ip.x;
            data[idx] = T; // std::min(T, data[idx]);
        }
    }
};

#endif // MRAY_SCATTER_GRID_H
