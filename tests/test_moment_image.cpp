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
#include <gtest/gtest.h>

#include "moment_image.h"
#include "moment_image_io.h"
#include "moment_quantization.h"

TEST(test_moment_image, test_compact)
{
    MomentImageHost mi(8, 4, 10, true);

    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 8; ++x)
        {
            mi.index[y * mi.width + x] = x;

            for (int l = 0; l < x; ++l)
                mi.data[mi.get_idx(x, y) + l] = x / 8.f;
        }
    }

    mi.compact();

    int moment_sum = 0;
    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 8; ++x)
        {
            moment_sum += mi.get_num_moments(x, y);
            ASSERT_EQ(mi.get_num_moments(x, y), x);
        }
    }

    ASSERT_EQ(mi.data.size(), moment_sum);

    ASSERT_EQ(mi.get_idx(1, 0), 0);
    ASSERT_EQ(mi.get_idx(2, 0), 1);
    ASSERT_EQ(mi.get_idx(3, 0), 3);
    ASSERT_EQ(mi.get_idx(4, 0), 6);

    write_moment_image(mi, "test.bin", false, 32);

    auto loaded_mi = load_moment_image("test.bin");

    ASSERT_EQ(loaded_mi.width, mi.width);
    ASSERT_EQ(loaded_mi.is_compact, mi.is_compact);
    ASSERT_EQ(loaded_mi.data.size(), mi.data.size());

    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 8; ++x)
            ASSERT_EQ(loaded_mi.get_num_moments(x, y), x);

    ASSERT_EQ(loaded_mi.get_idx(4, 0), 6);

    write_moment_image(mi, "test.bin", true, 20);
    auto comp_mi = load_moment_image("test.bin");

    ASSERT_EQ(comp_mi.width, mi.width);
    ASSERT_EQ(comp_mi.is_compact, mi.is_compact);
    ASSERT_EQ(comp_mi.data.size(), mi.data.size());

    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 8; ++x)
            ASSERT_EQ(comp_mi.get_num_moments(x, y), x);

    ASSERT_EQ(comp_mi.get_idx(4, 0), 6);

    static constexpr float eps = 1e-5f;
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 8; ++x)
            for (int l = 0; l < comp_mi.get_num_moments(x, y); ++l)
                ASSERT_NEAR(comp_mi.data[comp_mi.get_idx(x, y) + l], comp_mi.data[mi.get_idx(x, y) + l], eps);
}

TEST(test_moment_image, test_prediction_coding)
{
    MomentImageHost mi(8, 4, 10, true);

    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 8; ++x)
        {
            mi.index[y * mi.width + x] = x > 0 ? x : 0;

            for (int l = 0; l < x; ++l)
                mi.data[mi.get_idx(x, y) + l] = x / 8.f * 4.f - 1.f;
        }
    }

    mi.compact();
    mi.prediction_encode(0 /* Cauchy transformation */, 1e-7f /*bias*/);

    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 8; ++x)
            ASSERT_EQ(mi.get_num_moments(x, y), x);

    write_moment_image(mi, "test.bin", false, 32);

    auto loaded_mi = load_moment_image("test.bin");

    ASSERT_EQ(loaded_mi.width, mi.width);
    ASSERT_EQ(loaded_mi.is_compact, mi.is_compact);
    ASSERT_EQ(loaded_mi.prediction_code, mi.prediction_code);
    ASSERT_EQ(loaded_mi.data.size(), mi.data.size());

    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 8; ++x)
            ASSERT_EQ(loaded_mi.get_num_moments(x, y), x);

    write_moment_image(mi, "test.bin", true, 12);
    auto comp_mi = load_moment_image("test.bin");

    ASSERT_EQ(comp_mi.width, mi.width);
    ASSERT_EQ(comp_mi.is_compact, mi.is_compact);
    ASSERT_EQ(comp_mi.prediction_code, mi.prediction_code);
    ASSERT_EQ(comp_mi.data.size(), mi.data.size());

    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 8; ++x)
            ASSERT_EQ(comp_mi.get_num_moments(x, y), x);

    static constexpr float eps = 1e-5f;
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 8; ++x)
            for (int l = 0; l < comp_mi.get_num_moments(x, y) + 1; ++l)
                ASSERT_NEAR(comp_mi.data[comp_mi.get_idx(x, y) + l], comp_mi.data[mi.get_idx(x, y) + l], eps);
}

TEST(test_moment_image, test_image_quantization_prediction_coding)
{
    MomentImageHost mi(8, 4, 10, true);

    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 8; ++x)
        {
            mi.index[y * mi.width + x] = x > 0 ? x + 1 : 0;

            for (int l = 0; l < x; ++l)
                mi.data[mi.get_idx(x, y) + l] = x / 8.f;
        }
    }

    mi.compact();
    mi.prediction_code = true;
    // mi.prediction_encode();

    auto qs = moment_quantization::quantize_prediction_coding(mi, 16);

    moment_quantization::dequantize_prediction_coding(mi, 16, qs);

    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 8; ++x)
        {
            for (int l = 0; l < x; ++l)
                ASSERT_NEAR(mi.data[mi.get_idx(x, y) + l], x / 8.f, 1e-2f);
        }
    }
}

TEST(test_moment_image, test_quantization_table)
{
    MomentImageHost mi(8, 4, 10, true);
    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 8; ++x)
        {
            mi.index[y * mi.width + x] = x > 0 ? x + 1 : 0;

            for (int l = 0; l < x; ++l)
                mi.data[mi.get_idx(x, y) + l] = x / 8.f;
        }
    }

    mi.compact();
    mi.prediction_code = true;
    mi.coding_warp = CODING_WARP_NONE;

    vector<Byte> table(mi.num_moments);
    table[0] = 16;
    for (int i = 1; i < mi.num_moments; ++i)
        table[i] = 8 + i;
    moment_quantization::set_prediction_coding_quantization_table(table);

    write_moment_image(mi, "test_mi.bin", false, 16);

    auto loaded_mi = load_moment_image("test_mi.bin");

    auto loaded_table = moment_quantization::get_prediction_coding_quantization_table();

    ASSERT_EQ(loaded_table[0], table[0]);
    for (int i = 1; i < mi.num_moments; ++i)
        ASSERT_EQ(loaded_table[i], table[i]);

    ASSERT_EQ(mi.data.size(), loaded_mi.data.size());
    for (size_t i = 0; i < mi.data.size(); ++i)
        ASSERT_NEAR(mi.data[i], loaded_mi.data[i], 1e-2f);
}

MomentImageHost get_image(int t, int num_t)
{
    MomentImageHost mi(8, 4, 10, true);

    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 8; ++x)
        {
            mi.index[y * mi.width + x] = x > 0 ? x : 0;
            for (int l = 0; l < x; ++l)
                mi.data[mi.get_idx(x, y) + l] = x * t / static_cast<float>(num_t);
        }
    }
    mi.compact();
    return mi;
}

void test_moment_image_eq(const MomentImageHost &a, const MomentImageHost &b)
{
    ASSERT_EQ(a.width, b.width);
    ASSERT_EQ(a.height, b.height);
    ASSERT_EQ(a.index.size(), b.index.size());
    ASSERT_EQ(a.bounds.size(), b.bounds.size());

    for (size_t i = 0; i < a.index.size(); ++i)
        ASSERT_EQ(a.index[i], b.index[i]);

    for (int y = 0; y < a.height; ++y)
    {
        for (int x = 0; x < a.width; ++x)
        {
            for (int m = 0; m < a.get_num_moments(x, y); ++m)
                ASSERT_EQ(a.data[a.get_idx(x, y) + m], b.data[b.get_idx(x, y) + m]);
        }
    }
}

TEST(test_moment_image, test_io)
{
    MomentImageHost mi(8, 8, 10, true, true);

    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 8; ++x)
        {
            mi.index[y * mi.width + x] = x;

            for (int l = 0; l < x; ++l)
                mi.data[mi.get_idx(x, y) + l] = x / 8.f;
        }
    }

    mi.compact();

    write_moment_image(mi, "test.bin", false, 32);

    auto mi2 = load_moment_image("test.bin");

    test_moment_image_eq(mi, mi2);
}

TEST(test_moment_image, test_entropy_coding)
{
    MomentImageHost mi(8, 4, 10, false, true);
    mi.coding_warp = CODING_WARP_NONE;

    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 8; ++x)
        {
            for (int l = 0; l < mi.num_moments; ++l)
                mi.data[mi.get_idx(x, y) + l] = x / 8.f;
        }
    }

    write_moment_image(mi, "test.bin", false, 8, true);

    auto mi2 = load_moment_image("test.bin");

    test_moment_image_eq(mi, mi2);
}
