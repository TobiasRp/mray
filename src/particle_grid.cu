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
#include "particle_grid.cuh"

#include "cut/cuda_math.h"
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

using namespace cut;

/**
 * For all grid indices in cellPtr, this assigns every grid cell an index to the first particle in the cell.
 */
__global__ void setGridCellsKernel(GridCell *cells, const CellIndex *cellPtr, uint32_t numParticles)
{
    auto gId = blockDim.x * blockIdx.x + threadIdx.x;
    if (gId >= numParticles)
        return;

    if (gId == 0 || cellPtr[gId] != cellPtr[gId - 1])
        cells[cellPtr[gId]] = gId;
}

void setGridCellsGPU(GridCell *cells, const CellIndex *cellPtr, uint32_t num_particles)
{
    dim3 threadsPerBlock = 128;
    dim3 numPartBlocks = (num_particles + threadsPerBlock.x - 1) / threadsPerBlock.x;

    setGridCellsKernel<<<numPartBlocks, threadsPerBlock>>>(cells, cellPtr, num_particles);
}

unique_ptr<UniformGridGPU> createGridFromDomainGPU(const float *x, const float *y, const float *z, uint32_t num_particles, float r)
{
    thrust::device_ptr<const float> ptrX(x);
    thrust::device_ptr<const float> ptrY(y);
    thrust::device_ptr<const float> ptrZ(z);

    auto itX = thrust::minmax_element(ptrX, ptrX + num_particles);
    auto itY = thrust::minmax_element(ptrY, ptrY + num_particles);
    auto itZ = thrust::minmax_element(ptrZ, ptrZ + num_particles);

    Vec3f min(*itX.first, *itY.first, *itZ.first);
    Vec3f max(*itX.second, *itY.second, *itZ.second);

    return createGridFromDomain<true>(r, min, max);
}
