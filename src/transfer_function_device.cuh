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
#ifndef MRAY_TRANSFER_FUNCTION_DEVICE_CUH
#define MRAY_TRANSFER_FUNCTION_DEVICE_CUH

#include "transfer_function.h"

struct TransferFunctionDevice
{
    cudaArray_t data;
    cudaTextureObject_t tex;

    TransferFunctionDevice() = default;
    TransferFunctionDevice(const TransferFunction &tf)
    {
        cudaChannelFormatDesc formatDesc{32, 32, 32, 32, cudaChannelFormatKindFloat};

        CHECK_CUDA(cudaMallocArray(&data, &formatDesc, TransferFunction::RESOLUTION, TransferFunction::RESOLUTION));

        auto width = TransferFunction::RESOLUTION * 4 * sizeof(float);
        CHECK_CUDA(cudaMemcpy2DToArray(data, 0, 0, tf.data.data(), width /* pitch */, width, TransferFunction::RESOLUTION, cudaMemcpyHostToDevice));

        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = data;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.normalizedCoords = 1;

        CHECK_CUDA(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
    }

    __device__ Vec4f sample(float fs, float bs) const
    {
        return cut::make_vec4f(tex2D<float4>(tex, fs, bs));
    }
};

#endif // MRAY_TRANSFER_FUNCTION_DEVICE_CUH
