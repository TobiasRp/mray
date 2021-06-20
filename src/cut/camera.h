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
#ifndef MRAY_CAMERA_H
#define MRAY_CAMERA_H

#include "cuda_common.h"
#include "matrix.h"
#include <string>

namespace cut
{

/**
 * Both a simple perspective and orthograpic camera.
 * This makes the code quite confusing! Beware!
 */
struct Camera
{
    // For both!
    Mat4f transform;
    Vec3f position;

    // For ortho. only!
    Vec3f right, up, dir;

    // For persp. only!
    float scale;
    float aspect;
    float znear;
    bool orthographic;

    Mat4f inv_view_transform;

    FUNC Mat4f compute_inv_view_transform() const
    {
        assert(!orthographic);
        Mat3f R(transform[0][0], transform[0][1], transform[0][2],
                 transform[1][0], transform[1][1], transform[1][2],
                 transform[2][0], transform[2][1], transform[2][2]);
        auto Rinv = cut::transpose(R);

        Mat4f p_inv (1, 0, 0, -position.x,
                     0, 1, 0, -position.y,
                     0, 0, 1, -position.z,
                     0, 0, 0, 1.f);

        Mat4f R_inv(Rinv[0][0], Rinv[0][1], Rinv[0][2], 0,
                     Rinv[1][0], Rinv[1][1], Rinv[1][2], 0,
                     Rinv[2][0], Rinv[2][1], Rinv[2][2], 0,
                     0, 0, 0, 1);
        return R_inv * p_inv;
    }
};

extern cut::Camera load_from_file(std::string file, int width, int height);

} // namespace cut

#endif // MRAY_CAMERA_H
