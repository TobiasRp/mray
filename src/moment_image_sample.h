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
#ifndef MRAY_MOMENT_IMAGE_SAMPLE_H
#define MRAY_MOMENT_IMAGE_SAMPLE_H

#include "moment_image.h"

/**
 * Returns a subset of a moment image.
 */
inline MomentImageHost sample_moment_image(const MomentImageHost &img, int w, int h)
{
    int factor_x = img.width / w;
    int factor_y = img.height / h;

    MomentImageHost simg(w, h, img.num_moments, !img.index.empty());
    simg.prediction_code = img.prediction_code;
    simg.coding_warp = img.coding_warp;
    simg.coding_params = img.coding_params;

    for (int y = 0; y < w; ++y)
    {
        for (int x = 0; x < h; ++x)
        {
            int img_x = x * factor_x;
            int img_y = y * factor_y;

            auto num_moments = img.get_num_moments(img_x, img_y);
            simg.index[y * w + x] = num_moments;

            auto img_idx = img.get_idx(img_x, img_y);
            auto idx = simg.get_idx(x, y);

            for (int l = 0; l < num_moments; ++l)
                simg.data[idx + l] = img.data[img_idx + l];
        }
    }
    simg.compact();
    return simg;
}

#endif // MRAY_MOMENT_IMAGE_SAMPLE_H
