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
#ifndef MRAY_SAMPLES_IMAGE_DEVICE_H
#define MRAY_SAMPLES_IMAGE_DEVICE_H

#include "samples_image.h"

struct SamplesImageDeviceView
{
    int width, height;
    Sample *data;

    __device__ int get_idx(int x, int y) const { return (y * width + x) * SamplesImageHost::MAX_SAMPLES; }
};

struct SamplesImageDevice
{
    SamplesImageDeviceView view;

    cut::dev_ptr<Sample> data;

    SamplesImageDevice(const SamplesImageHost &img)
        : data(img.width * img.height * SamplesImageHost::MAX_SAMPLES)
    {
        view.width = img.width;
        view.height = img.height;
        view.data = data.get();
    }

    void copy_back(SamplesImageHost &img) const
    {
        data.copyToHost(img.data.data(), img.data.size());
    }
};

#endif // MRAY_SAMPLES_IMAGE_DEVICE_H
