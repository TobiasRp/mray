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
#include "moment_image_interpolator.h"

MomentImageHost create_test_image(float value, int num_offset, bool fix_num = false)
{
    MomentImageHost mi(8, 4, 10, true, true);

    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 8; ++x)
        {
            int num_moments = fix_num ? num_offset : x + num_offset;
            mi.index[y * mi.width + x] = num_moments;

            mi.set_bounds(x,y, DensityBound(0.0f, value));

            for (int l = 0; l < num_moments; ++l)
                mi.data[mi.get_idx(x, y) + l] = value;
        }
    }

    mi.compact();

    mi.add_error_bounds();
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 8; ++x)
            mi.set_error_bound(x, y, value);

    return mi;
}

TEST(moment_image_interpolator, test_temporal_interpolation)
{
    auto start = create_test_image(0.0f, 0);
    auto end = create_test_image(1.0f, 1);

    MomentImageInterpolator interp(start, end);

    auto half = interp.get(0.5f);

    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 8; ++x)
        {
            ASSERT_NEAR(half.get_bounds(x, y).y, 0.5f, 0.01f);
            (half.get_bounds(x, y).x, 0.0f);
            ASSERT_NEAR(half.get_bounds(x, y).y, 0.5f, 0.01f);
            (half.get_bounds(x, y).y, 0.5f);

            ASSERT_NEAR(half.get_bounds(x, y).y, 0.5f, 0.01f);

            ASSERT_NEAR(half.get_bounds(x, y).y, 0.5f, 0.01f);
            (half.get_error_bound(x, y), 0.5f);

            for (int l = 0; l < half.get_num_moments(x, y); ++l)
                ASSERT_EQ(half.data[half.get_idx(x, y) + l], 0.5f);
        }

    auto three_quarter = interp.get(0.75f);

    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 8; ++x)
            for (int l = 0; l < three_quarter.get_num_moments(x, y); ++l)
                ASSERT_EQ(three_quarter.data[three_quarter.get_idx(x, y) + l], 0.75f);
}

TEST(moment_image_interpolator, test_upsampling)
{
    auto img = create_test_image(1.0f, 2, true);

    auto big = MomentImageInterpolator::upsample(img, cut::Vec2i(2, 2));

    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 8; ++x)
        {
            ASSERT_EQ(big.get_bounds(x, y).x, 0.0f);
            ASSERT_EQ(big.get_bounds(x, y).y, 1.0f);

            ASSERT_EQ(big.get_error_bound(x, y), 1.0f);

            for (int l = 0; l < big.get_num_moments(x, y); ++l)
                ASSERT_EQ(big.data[big.get_idx(x, y) + l], 1.0f);
        }
    }
}

// TODO pred. coding test