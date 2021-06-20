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
#ifndef MOMENT_IO_H
#define MOMENT_IO_H

#include "common.h"
#include "coding_params.h"

using ErrorBound = uint16_t;

struct MomentImageHost
{
    int width, height;
    int num_moments; // Max num. of moments
    bool is_compact;
    bool prediction_code;
    CodingWarpType coding_warp;
    Range<Vec3f> domain; // AABB of the data

    vector<uint32_t> index;
    vector<DensityBound> bounds;
    vector<ErrorBound> error_bounds;
    vector<CodingParamType> coding_params;
    vector<float> data;

    MomentImageHost() = default;

    MomentImageHost(int w, int h, int num_moments, bool create_index, bool create_bounds = false)
        : width(w)
        , height(h)
        , num_moments(num_moments)
        , is_compact(false) // The image is never compact in the beginning!
        , prediction_code(false)
        , coding_warp(CODING_WARP_DEFAULT_TRANSFORMED)
    {
        if (create_index)
            index.resize(width * height + 1);
        if (create_bounds)
            bounds.resize(width * height);

        data.resize(get_elements_per_pixel() * width * height);
    }

    inline uint32_t get_elements_per_pixel() const
    {
        assert(!is_compact);
        return num_moments;
    }

    inline uint32_t get_idx_compact(int x, int y) const { return index[y * width + x]; }

    inline size_t get_idx(int x, int y) const
    {
        if (is_compact)
            return get_idx_compact(x, y);
        else
            return (y * width + x) * get_elements_per_pixel();
    }

    inline int get_num_moments(int x, int y) const
    {
        if (is_compact)
            return index[y * width + x + 1] - index[y * width + x];
        else
            return num_moments;
    }

    inline bool has_bounds() const { return !bounds.empty(); }
    inline void set_bounds(int x, int y, DensityBound bound) { bounds[y * width + x] = bound; }
    inline Vec2f get_bounds(int x, int y) const
    {
        if (has_bounds())
            return bounds[y * width + x].get();
        else
            return Vec2f(0.f, 1.f);
    }

    inline void add_error_bounds() { error_bounds.resize(width * height); }
    inline bool has_error_bounds() const { return !error_bounds.empty(); }
    inline void set_error_bound(int x, int y, float e)
    {
        assert(has_error_bounds());
        error_bounds[y * width + x] = static_cast<uint16_t>(e * MAX_UINT16);
    }
    inline float get_error_bound(int x, int y) const
    {
        assert(has_error_bounds());
        return static_cast<float>(error_bounds[y * width + x]) / MAX_UINT16;
    }

    inline bool has_coding_params() const { return !coding_params.empty(); }
    inline void set_coding_params(const vector<float> &params)
    {
        assert(coding_params.empty());
        assert(params.size() % 2 == 0);

        coding_params = CodingParamType::quantize(params);
    }
    inline vector<float> get_coding_params() const
    {
        return CodingParamType::dequantize(coding_params);
    }

    /**
     * Returns trigonometric moments at x, y. If the moment image is prediction coded, the
     * coding will be reversed.
     */
    void get_trig_moments(int x, int y, vector<float> &out) const;

    /**
     * Write the number of moments to >index<, then call this function
     * to compact the moment image.
     */
    void compact();

    /**
     * Perform prediction encoding for a compact moment image.
     */
    void prediction_encode(int warp, float bias);

    /**
    * Returns "prepared moments", i.e. Lagrange multipliers.
    */
    vector<float> prepare_moments(int x, int y, float bias) const;
};

#endif // MOMENT_IO_H
