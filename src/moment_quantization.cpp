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
#include "moment_quantization.h"

#include "moment_prediction_coding.h"
#include <fstream>

namespace moment_quantization
{

static bool s_UseQuantizationTable = false;
static array<Byte, 255> s_QuantizationTable;


uint32_t quantize(float value, int numBits, float min, float max)
{
    if (numBits == 32)
    {
        value = cut::clamp(value, min, max);
        uint32_t uvalue;
        std::memcpy(&uvalue, &value, sizeof(float));
        return uvalue;
    }

    value = cut::clamp(value, min, max);
    float norm = (value - min) / (max - min);

    uint32_t bits = norm * ((1 << numBits) - 1);
    uint32_t mask = 0xFFFFFFFF & ((1 << numBits) - 1);
    return bits & mask;
}

float dequantize(uint32_t q, int numBits, float min, float max)
{
    if (numBits == 32)
    {
        float fvalue;
        std::memcpy(&fvalue, &q, sizeof(float));
        return cut::clamp(fvalue, min, max);
    }

    float norm = q / static_cast<float>((1 << numBits) - 1);
    return norm * (max - min) + min;
}

inline void pack_bits(vector<Byte> &out, const vector<uint32_t> &sparse_bits, int num_bits)
{
    Byte buffer = 0;
    int current_bit = 0;

    for (size_t i = 0; i < sparse_bits.size(); ++i)
    {
        for (int b = 0; b < num_bits; ++b)
        {
            int bit = sparse_bits[i] & (1 << b);
            if (bit)
                buffer |= (1 << current_bit);

            ++current_bit;
            if (current_bit == 8)
            {
                out.push_back(buffer);
                buffer = 0;
                current_bit = 0;
            }
        }
    }
}

inline void pack_bits(vector<Byte> &out, const vector<uint32_t> &sparse_bits, const vector<Byte> &num_bits)
{
    Byte buffer = 0;
    int current_bit = 0;

    for (size_t i = 0; i < sparse_bits.size(); ++i)
    {
        for (int b = 0; b < num_bits[i]; ++b)
        {
            int bit = sparse_bits[i] & (1 << b);
            if (bit)
                buffer |= (1 << current_bit);

            ++current_bit;
            if (current_bit == 8)
            {
                out.push_back(buffer);
                buffer = 0;
                current_bit = 0;
            }
        }
    }
}

inline void unpack_bits(vector<uint32_t> &sparse_bits, const vector<Byte> &in, int num_bits)
{
    uint32_t buffer = 0;
    int current_bit = 0;

    for (size_t i = 0; i < in.size(); ++i)
    {
        for (int b = 0; b < 8; ++b)
        {
            int bit = in[i] & (1 << b);
            if (bit)
                buffer |= (1 << current_bit);

            ++current_bit;
            if (current_bit == num_bits)
            {
                sparse_bits.push_back(buffer);
                buffer = 0;
                current_bit = 0;
            }
        }
    }
}

inline void unpack_bits(vector<uint32_t> &sparse_bits, const vector<Byte> &in, const vector<Byte> &num_bits)
{
    uint32_t buffer = 0;
    int current_bit = 0;

    int num_idx = 0;
    for (size_t i = 0; i < in.size(); ++i)
    {
        for (int b = 0; b < 8; ++b)
        {
            int bit = in[i] & (1 << b);
            if (bit)
                buffer |= (1 << current_bit);

            ++current_bit;
            if (current_bit == num_bits[num_idx])
            {
                sparse_bits.push_back(buffer);
                buffer = 0;
                current_bit = 0;
                ++num_idx;
            }
        }
    }
}

bool uses_prediction_coding_quantization_table()
{
    return s_UseQuantizationTable;
}

void load_prediction_coding_quantization_table(const string &file)
{
    s_UseQuantizationTable = true;

    std::ifstream in(file, std::ifstream::in);

    string line;
    int i = 0;
    while (std::getline(in, line, ',') && i < 255)
    {
        s_QuantizationTable[i] = std::stoi(line);
        ++i;
    }
}

vector<Byte> get_prediction_coding_quantization_table(int max_moments, int numBits)
{
    vector<Byte> table(max_moments);
    for (int m = 0; m < max_moments; ++m)
    {
        if (s_UseQuantizationTable)
            table[m] = s_QuantizationTable[m];
        else
        {
            if (m == 0)
                table[m] = 16;
            else
                table[m] = numBits;
        }

    }
    return table;
}

void set_prediction_coding_quantization_table(const vector<Byte> &table)
{
    s_UseQuantizationTable = true;
    for (size_t i = 0; i < table.size(); ++i)
        s_QuantizationTable[i] = table[i];
}

vector<Byte> get_prediction_coding_quantization_table()
{
    vector<Byte> table(255, 32);
    if (s_UseQuantizationTable)
    {
        for (int i = 0; i < 255; ++i)
            table[i] = s_QuantizationTable[i];
    }
    else
        assert(false);
    return table;
}

void write_prediction_coding_quantization_table_to_file(const string &filename)
{
    std::fstream out(filename, std::fstream::out);

    for (size_t i = 0; i < s_QuantizationTable.size(); ++i)
    {
        if (i != s_QuantizationTable.size() -1)
            out << int(s_QuantizationTable[i]) << ", ";
        else
            out << int(s_QuantizationTable[i]);
    }
    out << "\n";
}

vector<Byte> quantize_prediction_coding(const MomentImageHost &mi, const vector<Byte> &quant_table)
{
    assert(mi.prediction_code);

    vector<uint32_t> words;
    vector<Byte> num_bits;

    for (int y = 0; y < mi.height; ++y)
    {
        for (int x = 0; x < mi.width; ++x)
        {
            auto num_moments = mi.get_num_moments(x, y);
            auto idx = mi.get_idx(x, y);

            if (num_moments == 0)
                continue;

            auto num_bits_0 = quant_table[0];
            auto ql = quantize(mi.data[idx], num_bits_0, 0.0f, 1.0f);
            words.push_back(ql);
            num_bits.push_back(num_bits_0);

            for (int l = 1; l < num_moments; ++l)
            {
                auto num_bits_l = quant_table[l];
                auto u_l = mi.data[idx + l];
                auto ql =
                    quantize(u_l, num_bits_l, PREDICTION_CODING_QUANTIZATION_MIN, PREDICTION_CODING_QUANTIZATION_MAX);
                words.push_back(ql);
                num_bits.push_back(num_bits_l);
            }
        }
    }

    vector<Byte> packed_bits;
    pack_bits(packed_bits, words, num_bits);
    return packed_bits;
}

vector<Byte> quantize_prediction_coding(const MomentImageHost &mi, int numBits)
{
    auto table = get_prediction_coding_quantization_table(mi.num_moments, numBits);
    return quantize_prediction_coding(mi, table);
}

void dequantize_prediction_coding(MomentImageHost &mi, const vector<Byte> &quant_table, const vector<Byte> &qs)
{
    assert(mi.prediction_code);

    vector<Byte> num_bits;
    for (int y = 0; y < mi.height; ++y)
    {
        for (int x = 0; x < mi.width; ++x)
        {
            auto num_moments = mi.get_num_moments(x, y);
            if (num_moments == 0)
                continue;

            for (int l = 0; l < num_moments; ++l)
                num_bits.push_back(quant_table[l]);
        }
    }

    vector<uint32_t> words;
    unpack_bits(words, qs, num_bits);

    size_t offset = 0;
    for (int y = 0; y < mi.height; ++y)
    {
        for (int x = 0; x < mi.width; ++x)
        {
            auto num_moments = mi.get_num_moments(x, y);
            auto idx = mi.get_idx(x, y);

            if (num_moments == 0)
                continue;

            uint32_t qj = words[offset];
            mi.data[idx] = dequantize(qj, num_bits[offset], 0.0, 1.0);
            assert(!std::isnan(mi.data[idx]) && !std::isinf(mi.data[idx]));
            ++offset;

            for (int l = 1; l < num_moments; ++l)
            {
                uint32_t qj = words[offset];
                mi.data[idx + l] = dequantize(qj, num_bits[offset], PREDICTION_CODING_QUANTIZATION_MIN,
                                              PREDICTION_CODING_QUANTIZATION_MAX);
                assert(!std::isnan(mi.data[idx + l]) && !std::isinf(mi.data[idx + l]));
                assert(mi.data[idx + l] >= -1.0f && mi.data[idx + l] <= 1.0f);
                ++offset;
            }
        }
    }
}

void dequantize_prediction_coding(MomentImageHost &mi, int numBits, const vector<Byte> &qs)
{
    auto table = get_prediction_coding_quantization_table(mi.num_moments, numBits);
    return dequantize_prediction_coding(mi, table, qs);
}

vector<Byte> quantize(const MomentImageHost &mi, int numBits)
{
    assert(!mi.prediction_code);

    vector<uint32_t> qs;
    for (int y = 0; y < mi.height; ++y)
    {
        for (int x = 0; x < mi.width; ++x)
        {
            auto num_moments = mi.get_num_moments(x, y);
            auto idx = mi.get_idx(x, y);

            if (num_moments == 0)
                continue;

            uint32_t q0 = quantize(mi.data[idx], numBits, 0.0f, 1.0f);
            qs.push_back(q0);

            for (int j = 1; j < num_moments; ++j)
            {
                float mj = mi.data[idx + j];
                auto qm = quantize(mj, numBits, -1.0 / M_PI, 1.0 / M_PI);
                qs.push_back(qm);
            }
        }
    }

    vector<Byte> packed_bits;
    pack_bits(packed_bits, qs, numBits);
    return packed_bits;
}

void dequantize(MomentImageHost &mi, int numBits, const vector<Byte> &qs)
{
    assert(!mi.prediction_code);

    vector<uint32_t> words;
    unpack_bits(words, qs, numBits);

    size_t offset = 0;
    for (int y = 0; y < mi.height; ++y)
    {
        for (int x = 0; x < mi.width; ++x)
        {
            auto num_moments = mi.get_num_moments(x, y);
            auto idx = mi.get_idx(x, y);

            if (num_moments == 0)
                continue;

            auto m0 = words[offset]; // read_bytes(qs, b_offset, num_bytes);
            ++offset;
            // b_offset += num_bytes;

            mi.data[idx] = dequantize(m0, numBits, 0.0f, 1.0f);
            assert(mi.data[idx] >= 0.0f / M_PI && mi.data[idx] <= 1.0f);

            for (int l = 1; l < num_moments; ++l)
            {
                uint32_t qj = words[offset]; // read_bytes(qs, b_offset, num_bytes);
                // b_offset += num_bytes;
                ++offset;
                mi.data[idx + l] = dequantize(qj, numBits, -1.0 / M_PI, 1.0 / M_PI);
                assert(mi.data[idx + l] >= -1.0f / M_PI && mi.data[idx + l] <= 1.0f / M_PI);
            }
        }
    }
}

void append(vector<Byte> &qs, uint32_t q, int numBits)
{
    int numBytes = (numBits + 7) / 8;

    if (numBytes == 1)
        qs.push_back(static_cast<Byte>(q));
    else if (numBytes == 2)
    {
        qs.push_back(static_cast<Byte>(q & 0xFF));
        qs.push_back(static_cast<Byte>((q >> 8) & 0xFF));
    }
    else if (numBytes == 3)
    {
        qs.push_back(static_cast<Byte>(q & 0xFF));
        qs.push_back(static_cast<Byte>((q >> 8) & 0xFF));
        qs.push_back(static_cast<Byte>((q >> 16) & 0xFF));
    }
    else
    {
        qs.push_back(static_cast<Byte>(q & 0xFF));
        qs.push_back(static_cast<Byte>((q >> 8) & 0xFF));
        qs.push_back(static_cast<Byte>((q >> 16) & 0xFF));
        qs.push_back(static_cast<Byte>((q >> 24) & 0xFF));
    }
}

uint32_t read_bits_at_idx(const vector<Byte> &qs, int idx, int numBits)
{
    int numBytes = (numBits + 7) / 8;

    if (numBytes == 1)
        return qs[idx];
    else if (numBytes == 2)
    {
        uint32_t res = qs[idx * numBytes];
        res |= qs[idx * numBytes + 1] << 8;
        return res;
    }
    else if (numBytes == 3)
    {
        uint32_t res = qs[idx * numBytes];
        res |= qs[idx * numBytes + 1] << 8;
        res |= qs[idx * numBytes + 2] << 16;
        return res;
    }
    else
    {
        uint32_t res = qs[idx * numBytes];
        res |= qs[idx * numBytes + 1] << 8;
        res |= qs[idx * numBytes + 2] << 16;
        res |= qs[idx * numBytes + 3] << 24;
        return res;
    }
}

vector<Byte> quantize(const vector<float> &moments, int numMoments, int numBits)
{
    int numRays = moments.size() / numMoments;

    vector<Byte> qs;
    qs.reserve(numRays * numMoments * static_cast<int>((numBits + 7) / 8));
    for (int i = 0; i < numRays; ++i)
    {
        float m0 = moments[i * numMoments];

        uint32_t q0 = quantize(m0, numBits, 0.0f, 1.0f);

        append(qs, q0, numBits);

        for (int j = 1; j < numMoments; ++j)
        {
            float mj = moments[i * numMoments + j];
            append(qs, quantize(mj, numBits, -1.0 / M_PI, 1.0 / M_PI), numBits);
        }
    }
    return qs;
}

vector<float> dequantize(const vector<Byte> &qs, int numMoments, int numBits)
{
    int numBytes = (numBits + 7) / 8;
    int numRays = qs.size() / numMoments / numBytes;

    vector<float> values;
    values.reserve(numRays * numMoments);
    for (int i = 0; i < numRays; ++i)
    {
        uint32_t q0 = read_bits_at_idx(qs, i * numMoments, numBits);

        values.push_back(dequantize(q0, numBits, 0.0f, 1.0f));

        for (int j = 1; j < numMoments; ++j)
        {
            uint32_t qj = read_bits_at_idx(qs, numMoments * i + j, numBits);
            values.push_back(dequantize(qj, numBits, -1.0 / M_PI, 1.0 / M_PI));
        }
    }
    return values;
}

} // namespace moment_quantization
