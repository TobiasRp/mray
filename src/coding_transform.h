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
#ifndef MRAY_CODING_TRANSFORM_H
#define MRAY_CODING_TRANSFORM_H

#include "moment_image.h"
#include <algorithm>

/**
 * Determine parameters for the coding transform, i.e. the min/max values per index to transform the moments.
 * 
 * @return The absolute of the minimal value |min(l)| at each index l*2
 * and the half range |max(l) - min(l) / 2| at each index l*2+1.
 */
inline vector<float> find_coding_transform_parameters(const MomentImageHost &img)
{
    vector<float> params((img.num_moments - 1) * 2, 0.0f);

#pragma omp parallel for default(none), shared(img, params)
    for (int l = 1; l < img.num_moments; ++l)
    {
        for (int y = 0; y < img.height; ++y)
        {
            for (int x = 0; x < img.width; ++x)
            {
                auto num_moments = img.get_num_moments(x, y);
                if (num_moments < l)
                    continue;

                auto idx = img.get_idx(x, y);
                auto value = cut::clamp(img.data[idx + l], -1.0f, 1.0f);

                params[(l - 1) * 2] = std::min(params[(l - 1) * 2], value);
                params[(l - 1) * 2 + 1] = std::max(params[(l - 1) * 2 + 1], value);
            }
        }
    }

    for (int pidx = 0; pidx < img.num_moments - 1; ++pidx)
    {
        params[pidx * 2 + 1] = cut::abs(params[pidx * 2 + 1] - params[pidx * 2]) * 0.5f;
        params[pidx * 2] = cut::abs(params[pidx * 2]);
    }

    return params;
}

#endif // MRAY_CODING_TRANSFORM_H
