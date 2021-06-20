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
#include "color_image.h"

#include <stdio.h>

void write_PPM(string file, const ColorImageHost &img, Vec3f background_color)
{
    FILE *fp = fopen(file.c_str(), "wb"); /* b - binary mode */
    (void) fprintf(fp, "P6\n%d %d\n255\n", img.width, img.height);

    for (int j = img.height - 1; j >= 0; --j)
    {
        for (int i = 0; i < img.width; ++i)
        {
            auto rgba = img.get_color(i, j);

            Vec4f color(rgba.r / 255.f, rgba.g / 255.f, rgba.b / 255.f, rgba.a / 255.f);

            Vec3f blended;
            blended.x = color.x * color.w + background_color.x * (1.f - color.w);
            blended.y = color.y * color.w + background_color.y * (1.f - color.w);
            blended.z = color.z * color.w + background_color.z * (1.f - color.w);

            Vec4f final(std::pow(blended.x, 1.f / 2.2f),
                        std::pow(blended.y, 1.f / 2.2f),
                        std::pow(blended.z, 1.f / 2.2f), 1.0f);

            ColorImageHost::rgba8_t final_rgba(final);
            (void) fwrite((unsigned char*) &final_rgba, 1, 3, fp);
        }
    }

    fclose(fp);
}
