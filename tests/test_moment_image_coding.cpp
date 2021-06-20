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

#include "moment_image_coding.h"
#include "moment_image.h"

TEST(test_moment_image_coding, test_coding)
{
    MomentImageHost mi(16, 16, 10, false, false);
    mi.coding_warp = CODING_WARP_NONE;
    mi.prediction_code = true;

    for (int y = 0; y < mi.height; ++y)
    {
        for (int x = 0; x < mi.width; ++x)
        {
            for (int l = 0; l < mi.num_moments; ++l)
                mi.data[mi.get_idx(x, y) + l] = x / static_cast<float>(mi.width);
        }
    }

    MomentImageHost mi_dec(16, 16, 10, false, false);
    mi_dec.coding_warp = CODING_WARP_NONE;
    mi_dec.prediction_code = true;

    vector<Byte> quant_table(mi.num_moments, 11);
    quant_table[0] = 16;
    auto bytes = entropy_encode(mi, quant_table);

    entropy_decode(mi_dec, quant_table, bytes);

    for (int y = 0; y < mi.height; ++y)
    {
        for (int x = 0; x < mi.height; ++x)
        {
            for (int l = 0; l < mi.num_moments; ++l)
            {
                ASSERT_NEAR(mi.data[mi.get_idx(x, y) + l], mi_dec.data[mi_dec.get_idx(x, y) + l], 0.1);
            }
        }
    }
}