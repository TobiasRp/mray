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
#include "cuda_memory.h"

#include <iostream>
#include <mutex>
#include <unordered_map>

namespace cut
{

struct AllocInfo
{
    size_t size;
    string name;
    bool unified;
};

static std::mutex s_allocLock;
static std::unordered_map<void *, AllocInfo> s_allocMap;

void *allocDevice(size_t size, const char *name)
{
    assert(size > 0);

    void *ptr;
    CHECK_CUDA(cudaMalloc(&ptr, size));

    s_allocLock.lock();
    s_allocMap[ptr] = AllocInfo{size, name, false};
    s_allocLock.unlock();

    return ptr;
}

void freeDevice(void *dptr)
{
    s_allocLock.lock();
    s_allocMap.erase(dptr);
    s_allocLock.unlock();

    CHECK_CUDA(cudaFree(dptr));
}

void *allocUnified(size_t size, const char *name)
{
    assert(size > 0);

    void *ptr;
    CHECK_CUDA(cudaMallocManaged(&ptr, size));

    s_allocLock.lock();
    s_allocMap[ptr] = AllocInfo{size, name, true};
    s_allocLock.unlock();

    return ptr;
}

void freeUnified(void *dptr)
{
    s_allocLock.lock();
    s_allocMap.erase(dptr);
    s_allocLock.unlock();

    CHECK_CUDA(cudaFree(dptr));
}

void reportUnreleasedDeviceMemory()
{
    s_allocLock.lock();
    for (auto it : s_allocMap)
    {
        std::cout << "Allocation from " << it.second.name << " not freed! " << it.second.size;
        if (it.second.unified)
            std::cout << " bytes of >unified memory<.\n";
        else
            std::cout << " bytes of >device memory<.\n";
    }
    s_allocLock.unlock();
}

void assertDeviceMemoryReleased()
{
    assert(s_allocMap.empty());
}

void reportDeviceMemory()
{
    size_t free, total;
    CHECK_CUDA(cudaMemGetInfo(&free, &total));

    free /= (1000 * 1000);
    total /= (1000 * 1000);

    std::cout << "Device memory: " << free << "mb / " << total << "mb" << std::endl;
}

size_t getAllocatedDeviceMemory()
{
    //    size_t free, total;
    //    CHECK_CUDA(cudaMemGetInfo(&free, &total));
    //    return total - free;

    size_t total = 0;
    s_allocLock.lock();
    for (auto it : s_allocMap)
    {
        total += it.second.size;
    }
    s_allocLock.unlock();
    return total;
}

} // namespace cut

#endif
