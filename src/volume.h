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
#ifndef MRAY_VOLUME_H
#define MRAY_VOLUME_H

#include "common.h"
#include "particle_grid.cuh"

struct RegularGrid
{
    enum class DataType
    {
        F32,
        U8
    };

    Vec3i size;
    Range<Vec3f> domain;
    DataType type;
    vector<float> data;

    RegularGrid(Vec3i size, Range<Vec3f> spatial_domain, DataType dtype)
        : size(size)
        , domain(spatial_domain)
        , type(dtype)
    {
    }

    RegularGrid(Vec3i size, Range<Vec3f> spatial_domain, const vector<float> &data, DataType dtype)
        : size(size)
        , domain(spatial_domain)
        , type(dtype)
        , data(data)
    {
    }

    inline float read(size_t idx) const
    {
        if (type == DataType::U8)
            return static_cast<float>(reinterpret_cast<const Byte *>(data.data())[idx] / 255.f);
        else if (type == DataType::F32)
            return reinterpret_cast<const float *>(data.data())[idx];
        else
            assert(false);
        return 0.0f;
    }

    inline int get_byte_size() const
    {
        if (type == DataType::U8)
            return 1;
        else if (type == DataType::F32)
            return 4;
        else
            assert(false);
        return 0;
    }

    inline bool is_normalized() const
    {
        if (type != DataType::F32)
            return true;
        else
            return false;
    }

    inline float get(int x, int y, int z) const
    {
        x = cut::clamp(x, 0, size.x - 1);
        y = cut::clamp(y, 0, size.y - 1);
        z = cut::clamp(z, 0, size.z - 1);
        return read(z * static_cast<size_t>(size.y * size.x) + y * size.x + x);
    }

    inline float sample(Vec3f p) const
    {
        Vec3f np = cut::div(p - domain.min, domain.max - domain.min);
        Vec3f ip = cut::mul(np, cut::cast<int, float>(size));

        Vec3f a = ip - Vec3f(std::floor(ip.x), std::floor(ip.y), std::floor(ip.z));

        int xl = std::floor(ip.x);
        int yl = std::floor(ip.y);
        int zl = std::floor(ip.z);

        // Trilinear interpolation
        auto lll = get(xl, yl, zl);
        auto lrl = get(xl, yl + 1, zl);
        auto rrl = get(xl + 1, yl + 1, zl);
        auto rll = get(xl + 1, yl, zl);

        auto llr = get(xl, yl, zl + 1);
        auto lrr = get(xl, yl + 1, zl + 1);
        auto rrr = get(xl + 1, yl + 1, zl + 1);
        auto rlr = get(xl + 1, yl, zl + 1);

        return (1.f - a.x) * (1.f - a.y) * (1.f - a.z) * lll + a.x * (1.f - a.y) * (1.f - a.z) * rll +
               (1.f - a.x) * a.y * (1.f - a.z) * lrl + a.x * a.y * (1.f - a.z) * rrl +
               (1.f - a.x) * (1.f - a.y) * a.z * llr + a.x * (1.f - a.y) * a.z * rlr + (1.f - a.x) * a.y * a.z * lrr +
               a.x * a.y * a.z * rrr;
    }

    float get_step_size() const
    {
        auto cells = cut::div(domain.max - domain.min, cut::make_vec3f(size));
        return cut::min3(cells.x, cells.y, cells.z);
    }
};

struct Particles
{
    Range<Vec3f> domain;

    vector<float> x, y, z, values;
    float smoothing_length;

    unique_ptr<UniformGridCPU> grid;

    Particles(Range<Vec3f> domain, const vector<float> &x, const vector<float> &y, const vector<float> &z,
              const vector<float> &values, float smoothing_length)
        : domain(domain)
        , x(x)
        , y(y)
        , z(z)
        , values(values)
        , smoothing_length(smoothing_length)
    {
        create_grid();
    }

    void create_grid();

    uint32_t size() const { return x.size(); }

    float sample(Vec3f p) const;

    float get_step_size() const { return smoothing_length * 0.5f; }
};

#endif // MRAY_VOLUME_H
