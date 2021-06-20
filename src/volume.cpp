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
#include "volume.h"
#include "cut/sort_util.h"
#include "particle_interpolate.h"

template <typename T>
void reorderingKernelCPU(const T *__restrict in, T *__restrict out, const uint32_t *indices, uint32_t num_particles)
{
    for (uint32_t gId = 0; gId < num_particles; ++gId)
    {
        out[gId] = in[indices[gId]];
    }
}

template <typename T, typename I>
inline void reorderByIndexHelper(const T *values, const I *indices, void *temp, uint32_t num_values)
{
    reorderingKernelCPU(values, static_cast<T *>(temp), indices, num_values);
    std::memcpy((void *)values, temp, sizeof(T) * num_values);
}

inline void calcParticleCellIndicesKernelCPU(UniformGridDevice &grid, const float *x, const float *y,
                                             const float *z, uint32_t num_particles, CellIndex *cellPtr)
{
#pragma omp parallel for default(none) shared(grid, num_particles, cellPtr, x, y, z)
    for (uint32_t gId = 0; gId < num_particles; ++gId)
    {
        cellPtr[gId] = grid.getCellIndex(cut::Vec3f{x[gId], y[gId], z[gId]});
    }
}

void Particles::create_grid()
{
    auto size = x.size();
    grid = createGridFromDomainCPU(x.data(), y.data(), z.data(), size, smoothing_length);

    vector<CellIndex> partCells(size);
    calcParticleCellIndicesKernelCPU(grid->d_grid, x.data(), y.data(), z.data(), size, partCells.data());

    vector<uint32_t> cell_indices;
    cut::getSortedIndices(partCells, cell_indices);

    vector<uint32_t> temp(size);
    reorderByIndexHelper(partCells.data(), cell_indices.data(), temp.data(), size);

    reorderByIndexHelper(x.data(), cell_indices.data(), temp.data(), size);
    reorderByIndexHelper(y.data(), cell_indices.data(), temp.data(), size);
    reorderByIndexHelper(z.data(), cell_indices.data(), temp.data(), size);
    reorderByIndexHelper(values.data(), cell_indices.data(), temp.data(), size);

    grid->allocCells();

    auto cells = grid->getCells();
    for (uint32_t i = 0; i < grid->getTotalSize(); ++i)
        cells[i] = EMPTY_CELL;

    setGridCellsCPU(grid->getCells(), partCells.data(), size);
}

float Particles::sample(Vec3f p) const
{
    InterpolateParticlesOp<true> op;
    op.p = p;
    op.values = values.data();
    op.inv_h = 1.f / smoothing_length;

    grid->d_grid.forEachNeighbor<true>(p, size(), x.data(), y.data(), z.data(), op);

    op.normalize();
    return op.value;
}