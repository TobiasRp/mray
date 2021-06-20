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
#ifndef MRAY_CODING_PARAMS_H
#define MRAY_CODING_PARAMS_H

using CodingWarpType = uint8_t;

/**
 * Type of transformation that is applied to the residuals after coding and before quantization.
 */
enum CodingWarp
{
    // Transforms the residuals using min/max values
    CODING_WARP_DEFAULT_TRANSFORMED = 0,

    // Does not transform the residuals
    CODING_WARP_NONE = 1,
};

FUNC bool requires_coding_parameters(CodingWarpType type)
{
    return type == CODING_WARP_DEFAULT_TRANSFORMED;
}

struct CodingParamType
{
    uint16_t param1;
    uint16_t param2;

    static float get_param1(CodingParamType c) { return (c.param1 / MAX_UINT16); }

    static float get_param2(CodingParamType c) { return (c.param2 / MAX_UINT16); }

    static vector<CodingParamType> quantize(const vector<float> &params)
    {
        vector<CodingParamType> res;
        for (size_t i = 0; i < params.size() / 2; ++i)
        {
            uint16_t p1 = static_cast<uint16_t>(std::ceil(cut::clamp(params[i * 2], 0.0f, 1.0f) * MAX_UINT16));
            uint16_t p2 = static_cast<uint16_t>(std::ceil(cut::clamp(params[i * 2 + 1], 0.0f, 1.0f) * MAX_UINT16));
            res.push_back(CodingParamType{p1, p2});
        }
        return res;
    }

    static vector<float> dequantize(const vector<CodingParamType> &coding_params)
    {
        vector<float> params;
        for (auto c : coding_params)
        {
            params.push_back(get_param1(c));
            params.push_back(get_param2(c));
        }
        return params;
    }
};

#endif // MRAY_CODING_PARAMS_H
