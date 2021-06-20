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

#include "coding_transform.h"
#ifdef CUDA_SUPPORT
#include "coding_transform.cuh"
#endif

MomentImageHost get_moment_image()
{
    int w = 16;
    int h = 1;
    MomentImageHost mi(w, h, 10, true);

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            mi.index[y * mi.width + x] = 10;

            for (int l = 0; l < 10; ++l)
                mi.data[mi.get_idx(x, y) + l] = x / static_cast<float>(w-1) - 0.6f;
        }
    }
    mi.compact();
    return mi;
}

TEST(test_coding_transform, test_host)
{
    auto mi =  get_moment_image();
    auto params = find_coding_transform_parameters(mi);

    for (int pidx = 0; pidx < mi.num_moments - 1; ++pidx)
    {
        ASSERT_NEAR(params[pidx*2], 0.6f, 1e-6f);
        ASSERT_NEAR(params[pidx*2+1], 0.5f, 1e-6f);
    }
}

#ifdef CUDA_SUPPORT
TEST(test_coding_transform, test_cuda)
{
    auto mi =  get_moment_image();
    auto params = find_coding_transform_parameters(mi);

    MomentImageDevice d_mi(mi);
    d_mi.load_from(mi);
    auto params_device = find_coding_transform_parameters_device(d_mi.view);

    for (int pidx = 0; pidx < mi.num_moments - 1; ++pidx)
    {
        ASSERT_NEAR(params[pidx*2], params_device[pidx*2], 1e-6f);
        ASSERT_NEAR(params[pidx*2+1], params_device[pidx*2+1], 1e-6f);
    }
}
#endif