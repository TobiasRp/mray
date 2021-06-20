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
#include "gtest/gtest.h"
#include "regular_grid_loader.h"
#include "particle_loader.h"

TEST(test_data_loading, grid_test)
{
    RegularGridLoader rgl("../../data/simple_volume.nc");
    ASSERT_EQ(rgl.num_steps(), 5);

    auto grid2 = rgl.get(2, "s");
    ASSERT_EQ(grid2.get(0, 0, 0), 2);
    ASSERT_EQ(grid2.sample(Vec3f(0.01f, 0.51f, 0.99f)), 2);
    ASSERT_EQ(grid2.sample(Vec3f(-1.1f, 1.1f, 2.1f)), 2);
}

TEST(test_data_loading, particle_test)
{
    ParticleLoader pl("../../data/ABC.h5part");

    ASSERT_EQ(pl.num_steps(), 10);

    auto p2 = pl.get(2, "u", 0.1f);
    ASSERT_EQ(p2.size(), 17576);
}