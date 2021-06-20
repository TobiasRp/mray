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
#include "moment_image_coding.h"
#include "moment_quantization.h"
#include "moment_prediction_coding.h"

#include "cut/timing.h"

// We're using the slower floating point version to use up to and including 2^14-bits
// An integer based version should be a bit faster
#include "entropy_coding/arithmetic_codec.h"

struct EncodedImage
{
    vector<uint16_t> zeroth;
    vector<vector<unsigned char>> codes;
};

EncodedImage read(const MomentImageHost &mi, const vector<Byte> &bytes)
{
    EncodedImage img;

    size_t off = 0;
    uint32_t size;
    std::memcpy(&size, &bytes[off], sizeof(uint32_t));
    off += sizeof(uint32_t);

    img.zeroth.resize(size);
    std::memcpy(img.zeroth.data(), &bytes[off], size * sizeof(uint16_t));
    off += size * sizeof(uint16_t);

    img.codes.resize(mi.num_moments - 1);
    for (size_t i = 0; i < img.codes.size(); ++i)
    {
        std::memcpy(&size, &bytes[off], sizeof(uint32_t));
        off += sizeof(uint32_t);

        img.codes[i].resize(size);
        std::memcpy(img.codes[i].data(), &bytes[off], size);
        off += size;
    }

    return img;
}

vector<Byte> write(const EncodedImage &img)
{
    vector<Byte> res;
    size_t total_size = sizeof(uint32_t) + img.zeroth.size() * sizeof(uint16_t);
    for (size_t i = 0; i < img.codes.size(); ++i)
        total_size += sizeof(uint32_t) + img.codes[i].size();
    res.resize(total_size);

    // Write zeroth
    uint32_t size = img.zeroth.size();
    size_t off = 0;
    std::memcpy(&res[off], &size, sizeof(uint32_t));
    off += sizeof(uint32_t);
    std::memcpy(&res[off], img.zeroth.data(), img.zeroth.size() * sizeof(uint16_t));
    off += img.zeroth.size() * sizeof(uint16_t);

    for (size_t i = 0; i < img.codes.size(); ++i)
    {
        size = img.codes[i].size();
        std::memcpy(&res[off], &size, sizeof(uint32_t));
        off += sizeof(uint32_t);

        std::memcpy(&res[off], img.codes[i].data(), img.codes[i].size());
        off += img.codes[i].size();
    }
    assert(off == total_size);
    return res;
}

template <typename DataModel>
EncodedImage ac_encode(const MomentImageHost &mi, const vector<Byte> &quant_table, vector<DataModel> &models)
{
    vector<Arithmetic_Codec> coders(models.size());
    for (size_t i = 0; i < coders.size(); ++i)
    {
        coders[i].set_buffer(sizeof(uint32_t) * mi.height * mi.width, nullptr);
        coders[i].start_encoder();
    }

    vector<uint16_t> zeroth_moment;
    zeroth_moment.reserve(mi.height * mi.width);

#pragma omp parallel for default(none) shared(mi, coders, quant_table, models, zeroth_moment)
    for (int l = 0; l < mi.num_moments; ++l)
    {
        for (int y = 0; y < mi.height; ++y)
        {
            for (int x = 0; x < mi.width; ++x)
            {
                auto num_moments = mi.get_num_moments(x, y);
                auto idx = mi.get_idx(x, y);

                if (num_moments <= l)
                    continue;

                if (l == 0)
                {
                    auto num_bits_0 = quant_table[0];
                    auto ql = moment_quantization::quantize(mi.data[idx], num_bits_0, 0.0f, 1.0f);
                    zeroth_moment.push_back(static_cast<uint16_t>(ql));
                }
                else
                {
                    auto num_bits_l = quant_table[l];
                    auto u_l = mi.data[idx + l];
                    uint32_t ql;
                    if (mi.prediction_code)
                        ql = moment_quantization::quantize(u_l, num_bits_l, PREDICTION_CODING_QUANTIZATION_MIN,
                                                           PREDICTION_CODING_QUANTIZATION_MAX);
                    else
                        ql = moment_quantization::quantize(u_l, num_bits_l, -1.0 / M_PI, 1.0 / M_PI);

                    coders[l - 1].encode(ql, models[l - 1]);
                }
            }
        }
    }

    EncodedImage img;
    img.zeroth = std::move(zeroth_moment);
    img.codes.resize(coders.size());
    for (size_t i = 0; i < coders.size(); ++i)
    {
        auto bytes = coders[i].stop_encoder();
        img.codes[i].resize(bytes);
        std::memcpy(img.codes[i].data(), coders[i].buffer(), bytes);
    }
    return img;
}

template <typename DataModel>
void ac_decode(MomentImageHost &mi, const vector<Byte> &quant_table, vector<DataModel> &models, const EncodedImage &img)
{
    vector<Arithmetic_Codec> coders(models.size());
    for (size_t i = 0; i < coders.size(); ++i)
    {
        coders[i].set_buffer(img.codes[i].size(), nullptr);

        std::memcpy(coders[i].buffer(), img.codes[i].data(), img.codes[i].size());

        coders[i].start_decoder();
    }

    size_t zeroth_offset = 0;
#pragma omp parallel for default(none) shared(mi, coders, quant_table, models, img, zeroth_offset)
    for (int l = 0; l < mi.num_moments; ++l)
    {
        for (int y = 0; y < mi.height; ++y)
        {
            for (int x = 0; x < mi.width; ++x)
            {
                auto num_moments = mi.get_num_moments(x, y);
                auto idx = mi.get_idx(x, y);

                if (num_moments <= l)
                    continue;

                if (l == 0)
                {
                    uint32_t qj = img.zeroth[zeroth_offset];
                    mi.data[idx] = moment_quantization::dequantize(qj, quant_table[0], 0.0, 1.0);
                    ++zeroth_offset;
                }
                else
                {
                    uint32_t qj = coders[l - 1].decode(models[l - 1]);

                    if (mi.prediction_code)
                        mi.data[idx + l] = moment_quantization::dequantize(
                            qj, quant_table[l], PREDICTION_CODING_QUANTIZATION_MIN, PREDICTION_CODING_QUANTIZATION_MAX);
                    else
                        mi.data[idx + l] = moment_quantization::dequantize(qj, quant_table[l], -1.0 / M_PI, 1.0 / M_PI);
                }
            }
        }
    }
}

vector<Byte> entropy_encode(const MomentImageHost &mi, const vector<Byte> &quant_table)
{
    SCOPED_CPU_QUERY("Entropy encoding");

    EncodedImage img;
    {
        vector<Adaptive_Data_Model> models(mi.num_moments - 1);
        for (int l = 1; l < mi.num_moments; ++l)
            models[l - 1].set_alphabet(1 << quant_table[l]);

        img = ac_encode(mi, quant_table, models);
    }

    assert(img.codes.size() == static_cast<size_t>(mi.num_moments - 1));
    return write(img);
}

void entropy_decode(MomentImageHost &mi, const vector<Byte> &quant_table, const vector<Byte> &bytes)
{
    SCOPED_CPU_QUERY("Entropy decoding");

    EncodedImage img = read(mi, bytes);
    {
        vector<Adaptive_Data_Model> models(mi.num_moments - 1);
        for (int l = 1; l < mi.num_moments; ++l)
            models[l - 1].set_alphabet(1 << quant_table[l]);

        ac_decode(mi, quant_table, models, img);
    }
}