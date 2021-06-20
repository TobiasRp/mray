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
#ifndef MRAY_MOMENT_IMAGE_DEVICE_H
#define MRAY_MOMENT_IMAGE_DEVICE_H

#include "common.h"
#include "moment_image.h"

struct MomentImageDeviceView
{
    int width, height;
    int num_moments;
    CodingWarpType coding_warp;

    bool bounded_density;
    bool is_compact;
    bool prediction_code;
    bool has_error_bounds;
    Range<Vec3f> domain;

    uint32_t *index;
    float *data;
    DensityBound *bounds;
    ErrorBound *error_bounds;

    __device__ inline size_t get_elements_per_pixel() const
    {
        assert(!is_compact);
        return num_moments;
    }

    __device__ inline uint32_t get_idx_compact(int x, int y) const { return index[y * width + x]; }

    __device__ inline uint32_t get_idx(int x, int y) const
    {
        if (is_compact)
            return get_idx_compact(x, y);
        else
            return (y * width + x) * get_elements_per_pixel();
    }

    __device__ inline int get_num_moments(int x, int y) const
    {
        if (is_compact)
            return index[y * width + x + 1] - index[y * width + x];
        else
            return num_moments;
    }

    __device__ inline bool has_bounds() const { return bounded_density; }

    __device__ inline Vec2f get_bounds(int x, int y) const
    {
        if (bounded_density)
            return bounds[y * width + x].get();
        else
            return Vec2f(0.f, 1.f);
    }

    __device__ inline void set_error_bound(int x, int y, float e)
    {
        error_bounds[y * width + x] = static_cast<uint16_t>(e * MAX_UINT16);
    }

    __device__ inline float get_error_bound(int x, int y) const { return error_bounds[y * width + x] / MAX_UINT16; }
};

struct MomentImageDevice
{
    MomentImageDeviceView view;

    cut::dev_ptr<uint32_t> index;
    cut::dev_ptr<float> data;
    cut::dev_ptr<DensityBound> bounds;
    cut::dev_ptr<ErrorBound> error_bounds;
    vector<CodingParamType> coding_params;
    size_t data_size;

    MomentImageDevice(const MomentImageHost &h_mi)
        : data(h_mi.data.size())
        , data_size(h_mi.data.size())
    {
        view.width = h_mi.width;
        view.height = h_mi.height;
        view.num_moments = h_mi.num_moments;
        view.coding_warp = h_mi.coding_warp;
        view.bounded_density = h_mi.has_bounds();
        view.is_compact = h_mi.is_compact;
        view.prediction_code = h_mi.prediction_code;
        view.has_error_bounds = h_mi.has_error_bounds();
        view.domain = h_mi.domain;
        view.data = data.get();

        if (!h_mi.index.empty())
        {
            index.resize(h_mi.index.size());
            view.index = index.get();
        }
        else
        {
            view.index = nullptr;
        }

        if (h_mi.has_bounds())
        {
            bounds.resize(h_mi.bounds.size());
            view.bounds = bounds.get();
        }
        else
            view.bounds = nullptr;

        if (h_mi.has_error_bounds())
        {
            error_bounds.resize(h_mi.bounds.size());
            view.error_bounds = error_bounds.get();
        }
        else
            view.error_bounds = nullptr;
    }

    /**
    * Initialize the device image from host memory.
    */
    void load_from(const MomentImageHost &h_mi)
    {
        if (h_mi.is_compact)
            index.loadFromHost(h_mi.index.data(), h_mi.index.size());
        if (h_mi.has_bounds())
            bounds.loadFromHost(h_mi.bounds.data(), h_mi.bounds.size());
        if (h_mi.has_error_bounds())
            error_bounds.loadFromHost(h_mi.error_bounds.data(), h_mi.error_bounds.size());
        coding_params = h_mi.coding_params;
        data.loadFromHost(h_mi.data.data(), h_mi.data.size());
    }

    /**
     * Copy the device image back to the host.
     */
    void copy_back(MomentImageHost &h_mi) const
    {
        if (!h_mi.index.empty())
            index.copyToHost(h_mi.index.data(), h_mi.index.size());

        if (h_mi.has_bounds())
            bounds.copyToHost(h_mi.bounds.data(), h_mi.bounds.size());

        h_mi.data.resize(data_size);
        data.copyToHost(h_mi.data.data(), h_mi.data.size());

        h_mi.is_compact = view.is_compact;
        h_mi.prediction_code = view.prediction_code;
        h_mi.coding_warp = view.coding_warp;
        h_mi.coding_params = coding_params;
    }

    void copy_back_errors(MomentImageHost &h_mi) const
    {
        assert(h_mi.has_error_bounds());
        error_bounds.copyToHost(h_mi.error_bounds.data(), h_mi.error_bounds.size());
    }

    int get_best_smem_launch_config(int bytes_per_moment, dim3 &num_blocks, dim3 &threads_per_block) const;

    /**
     * Compacts the height * width * max_moments image by removing unused moments (i.e. zero values).
     */
    void compact();

    /**
    * Perform prediction encoding for a compact moment image.
    */
    void prediction_encode(int coding_warp, float bias);

    /**
    * Revert prediction encoding back to bounded trigonometric moments.
    */
    void revert_prediction_coding();

    /**
     * Prepares the moments, i.e. computes Lagrange multipliers for efficient reconstruction.
     */
    void prepare_moments_device(float bias);

private:
    void load_coding_params();
};

#endif // MRAY_MOMENT_IMAGE_DEVICE_H
