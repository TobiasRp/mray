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
#include "moment_image_io.h"

#include "moment_quantization.h"
#include "moment_image_coding.h"
#include <fstream>
#include "lz4.h"

void read_bytes(size_t byte_size, Byte *out, bool compressed, std::fstream &in)
{
    if (compressed)
    {
        uint32_t compressed_size;
        in.read((char *)&compressed_size, sizeof(uint32_t));

        assert(compressed_size <= byte_size);

        if (compressed_size == byte_size)
        {
            // Failed to compress!
            in.read((char *)out, byte_size);
        }
        else
        {
            vector<Byte> compressed(compressed_size);
            in.read((char *)compressed.data(), compressed.size());

            auto r = LZ4_decompress_safe((const char *)compressed.data(), (char *)out, compressed.size(), byte_size);

            if (r < 0) // Failed to compress! Copy it!
                std::memcpy(out, compressed.data(), compressed.size());
        }
    }
    else
    {
        in.read((char *)out, byte_size);
    }
}

void read_moment_bytes(const MomentImageHost &mi, bool compressed, std::fstream &in, vector<Byte> &bytes)
{
    if (compressed)
    {
        uint32_t compressed_size;
        in.read((char *)&compressed_size, sizeof(uint32_t));

        auto size_bound = mi.index[mi.width * mi.height] * 4; // Upper bound due to prediction coding!
        bytes.resize(size_bound);

        if (compressed_size == size_bound || compressed_size == 0)
        {
            // Failed to compress!
            in.read((char *)bytes.data(), bytes.size());
        }
        else
        {
            vector<Byte> compressed(compressed_size);
            in.read((char *)compressed.data(), compressed.size());

            auto r = LZ4_decompress_safe((const char *)compressed.data(), (char *)bytes.data(), compressed.size(),
                                         bytes.size());

            if (r < 0) // Failed to compress! Simply copy it!
            {
                bytes.resize(compressed.size());
                std::memcpy(bytes.data(), compressed.data(), compressed.size());
            }
            else
            {
                // Resize to make sure the size is not an upper bound!
                bytes.resize(r);
            }
        }
    }
    else
    {
        auto startPos = in.tellg();
        in.seekg(0, std::ios::end);
        auto endPos = in.tellg();
        in.seekg(startPos);

        bytes.resize(endPos - startPos);
        in.read((char *)bytes.data(), bytes.size());
    }
}

void read_quantization_table(int max_moments, Byte compressed, std::fstream &in)
{
    vector<Byte> table(max_moments);
    read_bytes(sizeof(uint8_t) * max_moments, table.data(), compressed, in);
    moment_quantization::set_prediction_coding_quantization_table(table);
}

void read_index(MomentImageHost &mi, bool compressed, std::fstream &in)
{
    mi.index.resize(mi.width * mi.height + 1);

    vector<Byte> quantized_index(mi.width * mi.height);
    read_bytes(mi.width * mi.height, quantized_index.data(), compressed, in);

    // Dequantize index from 8-bit
    uint32_t sum = 0;
    for (size_t idx = 0; idx < mi.index.size() - 1; ++idx)
    {
        mi.index[idx] = sum;
        sum += quantized_index[idx];
    }
    mi.index[mi.width * mi.height] = sum;
}

void read_bounded_density(MomentImageHost &mi, bool compressed, std::fstream &in)
{
    mi.bounds.resize(mi.width * mi.height);
    read_bytes(mi.width * mi.height * sizeof(DensityBound), (Byte *)mi.bounds.data(), compressed, in);
}

void read_error_bounds(MomentImageHost &mi, bool compressed, std::fstream &in)
{
    mi.error_bounds.resize(mi.width * mi.height);
    read_bytes(mi.width * mi.height * sizeof(ErrorBound), (Byte *)mi.error_bounds.data(), compressed, in);
}

void read_coding_parameters(MomentImageHost &mi, bool compressed, std::fstream &in)
{
    mi.coding_params.resize(mi.num_moments - 1);
    read_bytes((mi.num_moments - 1) * sizeof(CodingParamType), (Byte *)mi.coding_params.data(), compressed, in);
}

void read_moment_image(MomentImageHost &mi, bool compressed, Byte quantization_bits, bool entropy_coding,
                       std::fstream &in)
{
    vector<Byte> bytes;
    read_moment_bytes(mi, compressed, in, bytes);

    mi.data.resize(mi.index[mi.width * mi.height]);
    if (entropy_coding)
    {
        if (mi.prediction_code)
            entropy_decode(mi, moment_quantization::get_prediction_coding_quantization_table(), bytes);
        else
        {
            vector<Byte> qtable(mi.num_moments, quantization_bits);
            entropy_decode(mi, qtable, bytes);
        }
    }
    else if (quantization_bits < 32)
    {
        if (mi.prediction_code)
            moment_quantization::dequantize_prediction_coding(mi, quantization_bits, bytes);
        else
            moment_quantization::dequantize(mi, quantization_bits, bytes);
    }
    else
    {
        std::memcpy(mi.data.data(), bytes.data(), bytes.size());
    }
}

void read_compact_image(MomentImageHost &mi, bool compressed, Byte quantization_bits, bool entropy_coding,
                        Byte bounded_density, Byte error_bounds, Byte coding_params,
                        std::fstream &in)
{
    read_index(mi, compressed, in);

    if (bounded_density)
        read_bounded_density(mi, compressed, in);

    if (error_bounds)
        read_error_bounds(mi, compressed, in);

    if (coding_params)
        read_coding_parameters(mi, compressed, in);

    read_moment_image(mi, compressed, quantization_bits, entropy_coding, in);
}

void read_static_image(MomentImageHost &mi, bool compressed, Byte quantization_bits, Byte bounded_density,
                       Byte error_bounds, Byte coding_params, std::fstream &in)
{
    assert(!mi.prediction_code);

    if (bounded_density)
        read_bounded_density(mi, compressed, in);

    if (error_bounds)
        read_error_bounds(mi, compressed, in);

    if (coding_params)
        read_coding_parameters(mi, compressed, in);

    int floatsPerPixel = mi.get_elements_per_pixel();
    size_t newSize = mi.width * mi.height * floatsPerPixel;

    if (compressed)
    {
        auto startPos = in.tellg();
        in.seekg(0, std::ios::end);
        auto endPos = in.tellg();
        in.seekg(startPos);

        vector<Byte> compressed(endPos - startPos);
        in.read((char *)compressed.data(), compressed.size());

        if (quantization_bits == 32)
        {
            mi.data.resize(newSize);
            int r = LZ4_decompress_safe((const char *)compressed.data(), (char *)mi.data.data(), compressed.size(),
                                        mi.data.size() * sizeof(float));
            assert(r == static_cast<int>(mi.data.size() * sizeof(float)));
            UNUSED(r);
        }
        else
        {
            vector<Byte> qs(newSize * static_cast<int>((quantization_bits + 7) / 8));

            int r =
                LZ4_decompress_safe((const char *)compressed.data(), (char *)qs.data(), compressed.size(), qs.size());
            assert(r == static_cast<int>(qs.size()));
            UNUSED(r);

            mi.data = moment_quantization::dequantize(qs, mi.num_moments, quantization_bits);
            assert(mi.data.size() == newSize);
        }
    }
    else
    {
        mi.data.resize(newSize);
        in.read((char *)mi.data.data(), mi.data.size() * sizeof(float));
    }
}

void read_header(MomentImageHost &mi, Byte &compressed, Byte &quantization_bits, Byte &entropy_coding,
                 Byte &bounded_density, Byte &error_bounds, Byte &coding_params, Byte &quant_table, std::fstream &in)
{
    uint8_t numMom;
    uint8_t coding_warp;
    uint8_t flags;
    uint16_t w, h;

    Vec3f min, max;
    in.read((char *)&w, sizeof(uint16_t));
    in.read((char *)&h, sizeof(uint16_t));

    in.read((char *)&numMom, sizeof(uint8_t));
    in.read((char *)&coding_warp, sizeof(uint8_t));
    in.read((char *)&flags, sizeof(uint8_t));
    in.read((char *)&quantization_bits, sizeof(uint8_t));

    entropy_coding = quantization_bits & (1 << 6);
    quant_table = quantization_bits & (1 << 7);
    quantization_bits = quantization_bits & 0x3F;

    compressed = flags & 1;
    mi.is_compact = flags & (1 << 1);
    mi.prediction_code = flags & (1 << 2);
    bounded_density = flags & (1 << 4);
    error_bounds = flags & (1 << 5);
    coding_params = flags & (1 << 7);

    in.read((char *)&min, sizeof(Vec3f));
    in.read((char *)&max, sizeof(Vec3f));

    mi.width = w;
    mi.height = h;
    mi.num_moments = numMom;
    mi.coding_warp = coding_warp;
    mi.domain = Range<Vec3f>(min, max);
}

MomentImageHost load_moment_image(const std::string filename)
{
    std::fstream in(filename, std::fstream::binary | std::fstream::in);

    MomentImageHost mi;

    Byte compressed, quantization_bits, entropy_coding, bounded_density, error_bounds, coding_params, quant_table;
    read_header(mi, compressed, quantization_bits, entropy_coding, bounded_density, error_bounds, coding_params,
                quant_table, in);

    if (quant_table)
        read_quantization_table(mi.num_moments, compressed, in);

    if (mi.is_compact)
        read_compact_image(mi, compressed, quantization_bits, entropy_coding, bounded_density, error_bounds,
                           coding_params, in);
    else
        read_static_image(mi, compressed, quantization_bits, bounded_density, error_bounds, coding_params, in);

    return mi;
}

void write_bytes(const Byte *data_ptr, uint32_t data_size, bool compress, std::fstream &out)
{
    if (compress)
    {
        vector<Byte> compressed(data_size);
        auto r = LZ4_compress_default((const char *)data_ptr, (char *)compressed.data(), data_size, compressed.size());

        if (r <= 0 || r >= static_cast<int>(compressed.size()))
        {
            uint32_t size = data_size;
            out.write((char *)&size, sizeof(uint32_t));
            out.write((char *)data_ptr, data_size);
        }
        else
        {
            uint32_t size = static_cast<uint32_t>(r);
            assert(size < std::numeric_limits<uint32_t>::max());
            out.write((char *)&size, sizeof(uint32_t));
            out.write((char *)compressed.data(), size);
        }
    }
    else
        out.write((char *)data_ptr, data_size);
}

void write_quantization_table(int max_moments, Byte compress, std::fstream &out)
{
    auto table = moment_quantization::get_prediction_coding_quantization_table();
    write_bytes(table.data(), sizeof(uint8_t) * max_moments, compress, out);
}

void write_index(const MomentImageHost &mi, bool compress, std::fstream &out)
{
    vector<Byte> quantized_index(mi.index.size() - 1);

    // Quantize index to 8-bit. Assumes a max of 255 moments!
    for (size_t idx = 0; idx < mi.index.size() - 1; ++idx)
    {
        auto num = mi.index[idx + 1] - mi.index[idx];
        assert(num < 256);
        quantized_index[idx] = static_cast<Byte>(num);
    }

    write_bytes(quantized_index.data(), quantized_index.size(), compress, out);
}

void write_bounded_density(const MomentImageHost &mi, bool compress, std::fstream &out)
{
    write_bytes((Byte *)mi.bounds.data(), mi.bounds.size() * sizeof(DensityBound), compress, out);
}

void write_error_bounds(const MomentImageHost &mi, bool compress, std::fstream &out)
{
    write_bytes((Byte *)mi.error_bounds.data(), mi.error_bounds.size() * sizeof(ErrorBound), compress, out);
}

void write_coding_params(const MomentImageHost &mi, bool compress, std::fstream &out)
{
    write_bytes((Byte *)mi.coding_params.data(), mi.coding_params.size() * sizeof(CodingParamType), compress, out);
}

void write_moment_image(const MomentImageHost &mi, bool compress, Byte quantizationBits, bool entropy_coding,
                        std::fstream &out)
{
    const Byte *data_ptr = reinterpret_cast<const Byte *>(mi.data.data());
    size_t data_size = mi.data.size() * sizeof(float);
    vector<Byte> quantized_data;
    if (entropy_coding)
    {
        if (mi.prediction_code)
        {
            quantized_data = entropy_encode(mi, moment_quantization::get_prediction_coding_quantization_table());
        }
        else
        {
            vector<Byte> qtable(mi.num_moments, quantizationBits);
            quantized_data = entropy_encode(mi, qtable);
        }

        data_ptr = quantized_data.data();
        data_size = quantized_data.size();
    }
    else if (mi.prediction_code && quantizationBits < 32)
    {
        quantized_data = moment_quantization::quantize_prediction_coding(mi, quantizationBits);
        data_ptr = quantized_data.data();
        data_size = quantized_data.size();
    }
    else if (!mi.prediction_code && quantizationBits < 32)
    {
        quantized_data = moment_quantization::quantize(mi, quantizationBits);
        data_ptr = quantized_data.data();
        data_size = quantized_data.size();
    }

    write_bytes(data_ptr, data_size, compress, out);
}

void write_static_image(const MomentImageHost &mi, bool compress, Byte quantizationBits, std::fstream &out)
{
    assert(!mi.prediction_code);

    if (mi.has_bounds())
        write_bounded_density(mi, compress, out);

    if (mi.has_error_bounds())
        write_error_bounds(mi, compress, out);

    if (mi.has_coding_params())
    {
        write_coding_params(mi, compress, out);
    }

    if (compress && quantizationBits < 32)
    {
        out.write((char *)&quantizationBits, sizeof(Byte));
        vector<Byte> data = moment_quantization::quantize(mi.data, mi.num_moments, quantizationBits);

        vector<Byte> compressed(data.size() * 4);
        auto size =
            LZ4_compress_default((const char *)data.data(), (char *)compressed.data(), data.size(), compressed.size());

        if (size <= 0 || size >= static_cast<int>(compressed.size()))
        {
            out.write((char *)data.data(), data.size());
        }
        else
        {
            out.write((char *)compressed.data(), size);
        }
    }
    else if (compress && quantizationBits == 32) // No quantization
    {
        vector<Byte> compressed(mi.data.size() * 4);
        auto size = LZ4_compress_default((const char *)mi.data.data(), (char *)compressed.data(),
                                         mi.data.size() * sizeof(float), compressed.size());

        if (size <= 0 || size >= static_cast<int>(compressed.size()))
        {
            out.write((char *)mi.data.data(), mi.data.size());
        }
        else
        {
            out.write((char *)compressed.data(), size);
        }
    }
    else
    {
        out.write((char *)mi.data.data(), mi.data.size() * sizeof(float));
    }
}

void write_compact_image(const MomentImageHost &mi, bool compress, Byte quantizationBits, bool entropy_coding,
                         std::fstream &out)
{
    write_index(mi, compress, out);

    if (mi.has_bounds())
        write_bounded_density(mi, compress, out);

    if (mi.has_error_bounds())
        write_error_bounds(mi, compress, out);

    if (mi.has_coding_params())
        write_coding_params(mi, compress, out);

    write_moment_image(mi, compress, quantizationBits, entropy_coding, out);
}

void write_header(const MomentImageHost &mi, bool compress, Byte quantizationBits, bool entropy_coding,
                  bool quant_table, std::fstream &out)
{
    auto w = static_cast<uint16_t>(mi.width);
    auto h = static_cast<uint16_t>(mi.height);
    auto numMom = static_cast<uint8_t>(mi.num_moments);
    auto coding_warp = static_cast<uint8_t>(mi.coding_warp);
    auto quantization_bits = static_cast<uint8_t>(quantizationBits);
    quantization_bits |= entropy_coding << 6;
    quantization_bits |= quant_table << 7;

    uint8_t flags = 0;
    flags |= compress;
    flags |= mi.is_compact << 1;
    flags |= mi.prediction_code << 2;
    flags |= mi.has_bounds() << 4;
    flags |= mi.has_error_bounds() << 5;
    flags |= mi.has_coding_params() << 7;

    out.write((char *)&w, sizeof(uint16_t));
    out.write((char *)&h, sizeof(uint16_t));

    out.write((char *)&numMom, sizeof(uint8_t));
    out.write((char *)&coding_warp, sizeof(uint8_t));
    out.write((char *)&flags, sizeof(uint8_t));
    out.write((char *)&quantization_bits, sizeof(uint8_t));

    out.write((char *)&mi.domain.min, sizeof(Vec3f));
    out.write((char *)&mi.domain.max, sizeof(Vec3f));
}

void write_moment_image(const MomentImageHost &mi, const std::string filename, bool compress, Byte quantizationBits,
                        bool entropy_coding)
{
    std::fstream out(filename, std::fstream::binary | std::fstream::out);

    auto quant_table = moment_quantization::uses_prediction_coding_quantization_table();
    write_header(mi, compress, quantizationBits, entropy_coding, quant_table, out);

    if (quant_table)
        write_quantization_table(mi.num_moments, compress, out);

    if (mi.is_compact)
        write_compact_image(mi, compress, quantizationBits, entropy_coding, out);
    else
        write_static_image(mi, compress, quantizationBits, out);
}