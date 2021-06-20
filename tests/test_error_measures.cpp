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

#include "error_measures.h"

TEST(test_error_measures, error_percentile)
{
    // Example from https://www.cse.wustl.edu/%7Ejain/papers/ftp/psqr.pdf

    ErrorPercentile ep(50);

    ep.update(0, 0.02f);
    ep.update(0, 0.15f); // Typo in the paper!
    ep.update(0, 0.74f);
    ep.update(0, 3.39f);
    ep.update(0, 0.83f);

    float eps = 0.01f;

    ep.update(0, 22.37);
    ASSERT_NEAR(ep.error(), 0.74, eps);

    ep.update(0, 10.15);
    ASSERT_NEAR(ep.error(), 0.74, eps);

    ep.update(0, 15.43);
    ASSERT_NEAR(ep.error(), 2.18, eps);

    ep.update(0, 38.62);
    ASSERT_NEAR(ep.error(), 4.75, eps);

    ep.update(0, 15.92);
    ASSERT_NEAR(ep.error(), 4.75, eps);

    ep.update(0, 34.60);
    ASSERT_NEAR(ep.error(), 9.28, eps);

    ep.update(0, 10.28);
    ASSERT_NEAR(ep.error(), 9.28, eps);

    ep.update(0, 1.47);
    ASSERT_NEAR(ep.error(), 9.28, eps);

    ep.update(0, 0.40);
    ASSERT_NEAR(ep.error(), 9.28, eps);

    ep.update(0, 0.05);
    ASSERT_NEAR(ep.error(), 6.3, eps);

    ep.update(0, 11.39);
    ASSERT_NEAR(ep.error(), 6.3, eps);

    ep.update(0, 0.27);
    ASSERT_NEAR(ep.error(), 6.3, eps);

    ep.update(0, 0.42);
    ASSERT_NEAR(ep.error(), 6.3, eps);

    ep.update(0, 0.09);
    ASSERT_NEAR(ep.error(), 4.44, eps);

    ep.update(0, 11.37);
    ASSERT_NEAR(ep.error(), 4.44, eps);
}