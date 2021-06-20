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
#ifndef MRAY_SINGLE_SCATTERING_CACHE_DEVICE_CUH
#define MRAY_SINGLE_SCATTERING_CACHE_DEVICE_CUH

#include "single_scattering_image.h"
#include "single_scattering_cache.h"
#include "moment_image_device.cuh"
#include "volume_device.cuh"

struct SingleScatteringCacheDeviceView
{
    RegularGridDeviceView cache;

    FUNC bool enabled() const { return true; }
    FUNC bool use_cache() const { return true; }

    __device__ float sample_cache(Vec3f pt) const { return cache.sample(pt); }

    __device__ void set_smem(strided_array<float> mem) { UNUSED(mem); }

    FUNC int get_max_moments() const { return 0; }
};

template <> struct is_single_scattering_cached<SingleScatteringCacheDeviceView>
{
    static const bool value = true;
};

struct SingleScatteringCacheDevice
{
    SingleScatteringCacheDeviceView view;

    SingleScatteringCacheDevice(const RegularGridDeviceView &cacheview) { view.cache = cacheview; }

    SingleScatteringCacheDevice(const SingleScatteringCacheDevice &) = delete;
    SingleScatteringCacheDevice &operator=(const SingleScatteringCacheDevice &) = delete;

    SingleScatteringCacheDevice(SingleScatteringCacheDevice &&) = default;
    SingleScatteringCacheDevice &operator=(SingleScatteringCacheDevice &&) = default;

    SingleScatteringCacheDeviceView get_view() { return view; }

    void prepare(float bias) { UNUSED(bias); }
};

#endif // MRAY_SINGLE_SCATTERING_CACHE_DEVICE_CUH
