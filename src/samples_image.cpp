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
#include "samples_image.h"

#include <fstream>

void compact_image(const SamplesImageHost &img, vector<size_t> &indices, vector<Sample> &data)
{
    indices.resize(img.width * img.height + 1);

    size_t idx_sum = 0;
    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            uint32_t num = 0;

            auto s_idx = img.get_idx(x, y);
            auto samples_xy = &img.data[s_idx];
            while (samples_xy[num] != INVALID_SAMPLE && num < img.MAX_SAMPLES)
                ++num;

            for (uint32_t m = 0; m < num; ++m)
                data.push_back(samples_xy[m]);

            indices[y * img.width + x] = idx_sum;
            idx_sum += num;
        }
    }
    indices[img.height * img.width] = idx_sum;
}

void write_samples_image(string filename, const SamplesImageHost &img, bool compact)
{
    std::fstream out(filename, std::fstream::binary | std::fstream::out);
    auto max_samples = static_cast<uint32_t>(SamplesImageHost::MAX_SAMPLES);
    auto w = static_cast<uint32_t>(img.width);
    auto h = static_cast<uint32_t>(img.height);
    auto c = static_cast<uint8_t>(compact);
    auto padding = static_cast<uint8_t>(0);
    out.write((char *)&max_samples, sizeof(uint32_t));
    out.write((char *)&w, sizeof(uint32_t));
    out.write((char *)&h, sizeof(uint32_t));
    out.write((char *)&c, sizeof(uint8_t));
    out.write((char *)&padding, sizeof(uint8_t));
    out.write((char *)&padding, sizeof(uint8_t));
    out.write((char *)&padding, sizeof(uint8_t));

    if (compact)
    {
        vector<size_t> indices;
        vector<Sample> data;
        compact_image(img, indices, data);

        out.write((char *)indices.data(), indices.size() * sizeof(size_t));
        out.write((char *)data.data(), data.size() * sizeof(Sample));
    }
    else
        out.write((char *)img.data.data(), img.data.size() * sizeof(Sample));
}