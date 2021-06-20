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
#include "extrema.h"
#include "extrema.cuh"

#include <algorithm>

namespace cut
{

void find_min(const float *values, size_t num_values, float &min)
{
#ifdef CUDA_SUPPORT
    cut::find_min_host(values, num_values, min);
#else
    min = values[0];

    for (size_t i = 0; i < num_values; ++i)
        min = std::min(min, values[i]);
#endif
}

void find_max(const float *values, size_t num_values, float &max)
{
#ifdef CUDA_SUPPORT
    cut::find_max_host(values, num_values, max);
#else
    max = values[0];

    for (size_t i = 0; i < num_values; ++i)
        max = std::max(max, values[i]);
#endif
}

void find_minmax(const float *values, size_t num_values, float &min, float &max)
{
#ifdef CUDA_SUPPORT
    cut::find_minmax_host(values, num_values, min, max);
#else
    min = values[0];
    max = values[0];

    for (size_t i = 0; i < num_values; ++i)
    {
        min = std::min(min, values[i]);
        max = std::max(max, values[i]);
    }
#endif
}

} // namespace cut