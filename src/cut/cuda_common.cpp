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
#ifdef CUDA_SUPPORT
#include "cuda_common.h"

#include <iostream>

cudaDeviceProp g_DeviceProp;

void checkCUDA(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        if (err == cudaErrorMemoryAllocation)
        {
            std::cerr << "CUDA ERROR: Failed to allocate device memory (Out of memory, will abort)\n";
            std::cerr << "Reported device memory consumption:\n";
            cut::reportDeviceMemory();

            std::cerr << "Allocated device memory:\n";
            cut::reportUnreleasedDeviceMemory();

            std::terminate();
        }
        else
        {
            std::cerr << "CUDA ERROR: " << cudaGetErrorString(err) << " at " << file << ":" << line << std::endl;
            std::terminate();
        }
    }
}

void init_device(int dev_id)
{
    cudaSetDevice(dev_id);
    cudaGetDeviceProperties(&g_DeviceProp, dev_id);
}

#endif // CUDA_SUPPORT