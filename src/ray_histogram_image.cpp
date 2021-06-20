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
#include "ray_histogram_image.h"

#include <fstream>
#include "lz4.h"

void read_array(std::fstream &in, char *data, size_t byte_size, bool compressed)
{
    if (compressed)
    {
        uint32_t compressed_size;
        in.read((char *)&compressed_size, sizeof(uint32_t));

        if (compressed_size >= byte_size)
        {
            in.read(data, byte_size);
        }
        else
        {
            vector<Byte> compressed(compressed_size);
            in.read((char *)compressed.data(), compressed.size());

            auto r = LZ4_decompress_safe((const char *)compressed.data(), (char *)data, compressed.size(), byte_size);

            if (r < 0) // Failed to compress! Copy it!
                std::memcpy(data, compressed.data(), compressed.size());
        }
    }
    else
        in.read(data, byte_size);
}

void write_array(std::fstream &out, const char *data, size_t byte_size, bool compressed)
{
    if (compressed)
    {
        vector<char> compressed(byte_size);
        auto compressed_size = LZ4_compress_default(data, (char *)compressed.data(), byte_size, compressed.size());

        if (compressed_size <= 0 || compressed_size >= static_cast<int>(compressed.size()))
        {
            out.write((char *)&byte_size, sizeof(uint32_t));
            out.write(data, byte_size);
        }
        else
        {
            out.write((char *)&compressed_size, sizeof(uint32_t));
            out.write(compressed.data(), compressed_size);
        }
    }
    else
        out.write(data, byte_size);
}

RayHistogramImageHost read_rayhistogram_image(string filename)
{
    std::fstream in(filename, std::fstream::binary | std::fstream::in);

    uint16_t w, h, padding, compressed;
    uint32_t num_frusta, hist_size;
    in.read((char *)&w, sizeof(uint16_t));
    in.read((char *)&h, sizeof(uint16_t));
    in.read((char *)&padding, sizeof(uint16_t));
    in.read((char *)&compressed, sizeof(uint16_t));

    in.read((char *)&num_frusta, sizeof(uint32_t));
    in.read((char *)&hist_size, sizeof(uint32_t));

    Vec3f min, max;
    in.read((char *)&min, sizeof(Vec3f));
    in.read((char *)&max, sizeof(Vec3f));

    RayHistogramImageHost img(w, h);
    img.domain = Range<Vec3f>(min, max);
    img.indices.resize(w * h + 1);
    img.frusta.resize(num_frusta);
    img.frequencies.resize(hist_size);
    img.bin_ids.resize(hist_size);

    read_array(in, (char *)img.indices.data(), sizeof(uint32_t) * img.indices.size(), compressed);
    read_array(in, (char *)img.frusta.data(), sizeof(SubFrustum) * img.frusta.size(), compressed);
    read_array(in, (char *)img.frequencies.data(), sizeof(uint16_t) * img.frequencies.size(), compressed);
    read_array(in, (char *)img.bin_ids.data(), sizeof(uint16_t) * img.bin_ids.size(), compressed);

    return img;
}

void write_rayhistogram_image(string filename, const RayHistogramImageHost &img, bool compress)
{
    std::fstream out(filename, std::fstream::binary | std::fstream::out);
    auto w = static_cast<uint16_t>(img.width);
    auto h = static_cast<uint16_t>(img.height);
    auto padding = static_cast<uint16_t>(0);
    auto compression = static_cast<uint16_t>(compress);

    auto num_frusta = static_cast<uint32_t>(img.frusta.size());
    auto hist_size = static_cast<uint32_t>(img.frequencies.size());

    out.write((char *)&w, sizeof(uint16_t));
    out.write((char *)&h, sizeof(uint16_t));
    out.write((char *)&padding, sizeof(uint16_t));
    out.write((char *)&compression, sizeof(uint16_t));

    out.write((char *)&num_frusta, sizeof(uint32_t));
    out.write((char *)&hist_size, sizeof(uint32_t));

    out.write((char *)&img.domain.min, sizeof(Vec3f));
    out.write((char *)&img.domain.max, sizeof(Vec3f));

    write_array(out, (char *)img.indices.data(), sizeof(uint32_t) * img.indices.size(), compress);
    write_array(out, (char *)img.frusta.data(), sizeof(SubFrustum) * img.frusta.size(), compress);
    write_array(out, (char *)img.frequencies.data(), sizeof(uint16_t) * img.frequencies.size(), compress);
    write_array(out, (char *)img.bin_ids.data(), sizeof(uint16_t) * img.bin_ids.size(), compress);
}