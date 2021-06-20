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
#include "camera.h"

#include <fstream>
using namespace std;

cut::Camera cut::load_from_file(string file, int width, int height)
{
    std::ifstream fs(file, std::ios::in);
    cut::Camera cam;

    fs >> cam.orthographic;

    if (cam.orthographic)
    {
        // View-Projection
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 4; ++i)
                fs >> cam.transform[j][i];

        fs >> cam.position.x;
        fs >> cam.position.y;
        fs >> cam.position.z;

        fs >> cam.dir.x;
        fs >> cam.dir.y;
        fs >> cam.dir.z;

        fs >> cam.up.x;
        fs >> cam.up.y;
        fs >> cam.up.z;

        fs >> cam.right.x;
        fs >> cam.right.y;
        fs >> cam.right.z;
    }
    else
    {
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 4; ++i)
                fs >> cam.transform[j][i];

        cam.position = Vec3f(cam.transform[0][3], cam.transform[1][3], cam.transform[2][3]);

        fs >> cam.scale;
        fs >> cam.znear;

        cam.aspect = width / static_cast<float>(height);

        cam.inv_view_transform = cam.compute_inv_view_transform();
    }
    return cam;
}