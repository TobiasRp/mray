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
#ifndef PARTICLE_GRID_CUH
#define PARTICLE_GRID_CUH

#include "common.h"
#include "cut/cuda_math.h"
#include "cut/matrix.h"

#include <limits>

using GridCell = uint32_t;
using CellIndex = uint32_t;

#define EMPTY_CELL UINT32_MAX

//---------------------------------------------------------------------------------------------------------------

FUNC uint32_t getNextMultiple(uint32_t x, int multiple)
{
    return ((x - 1) / multiple + 1) * multiple;
}

struct UniformGridDevice
{
    float r_squared;
    cut::Vec3i tile_size;
    CellIndex total_size;
    cut::Vec3u size;
    cut::Vec3f min_pt;
    float inv_cell_size;

    GridCell *cells;

    FUNC cut::Vec3i getCell(cut::Vec3f p) const
    {
        return cut::Vec3i{static_cast<int>((p.x - min_pt.x) * inv_cell_size),
                          static_cast<int>((p.y - min_pt.y) * inv_cell_size),
                          static_cast<int>((p.z - min_pt.z) * inv_cell_size)};
    }

    FUNC CellIndex getIndexFromCell(cut::Vec3i c) const
    {
        int widthInTiles = (size.x + tile_size.x - 1) / tile_size.x;
        int heightInTiles = (size.y + tile_size.y - 1) / tile_size.y;

        int tileX = c.x / tile_size.x;
        int tileY = c.y / tile_size.y;
        int tileZ = c.z / tile_size.z;

        int inTileX = c.x % tile_size.x;
        int inTileY = c.y % tile_size.y;
        int inTileZ = c.z % tile_size.z;

        return (tileZ * heightInTiles * widthInTiles + tileY * widthInTiles + tileX) *
                   (tile_size.x * tile_size.y * tile_size.z) +
               inTileZ * tile_size.x * tile_size.y + inTileY * tile_size.x + inTileX;

        // Linear indexing
        //        return c.x + c.y * size.x + c.z * size.x * size.y;
    }

    FUNC CellIndex getCellIndex(cut::Vec3f p) { return getIndexFromCell(getCell(p)); }

    /**
     * Searches the neighborhood of the given point p of the timestep using an uniform grid.
     * A maximum of MAX_NEIGHBORS is written to out.
     */
    template <int MAX_NEIGHBORS, bool INCLUDE_ZERO_DIST = true, typename TPartSize>
    FUNC uint32_t neighborSearch(cut::Vec3f p, TPartSize num_particles, const float *d_x, const float *d_y,
                                 const float *d_z, TPartSize *out) const
    {
        cut::Vec3i cell = getCell(p);

        uint32_t numFound = 0;
        for (int z = cut::max(cell.z - 1, 0); z < cut::min(cell.z + 2, (int)size.z); ++z)
        {
            for (int y = cut::max(cell.y - 1, 0); y < cut::min(cell.y + 2, (int)size.y); ++y)
            {
                for (int x = cut::max(cell.x - 1, 0); x < cut::min(cell.x + 2, (int)size.x); ++x)
                {
                    auto nbCellIdx = getIndexFromCell(cut::Vec3i{x, y, z});
                    auto nbCellPtr = cells[nbCellIdx];

                    if (nbCellPtr == EMPTY_CELL)
                        continue;

                    // Find out how many particles are in the current cell
                    auto nIt = nbCellIdx + 1;
                    while (nIt < total_size && cells[nIt] == EMPTY_CELL)
                        ++nIt;

                    int partsInCell;
                    if (nIt == total_size)
                    {
                        partsInCell = num_particles - nbCellPtr;
                    }
                    else
                    {
                        assert(cells[nIt] != EMPTY_CELL);
                        auto cellsNit = cells[nIt];
                        partsInCell = cellsNit - nbCellPtr;
                    }

                    assert(partsInCell > 0);

                    // For every particle in the cell, check if it is inside the radius
                    for (int n = 0; n < partsInCell; ++n)
                    {
                        auto nbIdx = nbCellPtr + n;

                        assert(nbIdx < num_particles);

                        cut::Vec3f np{d_x[nbIdx], d_y[nbIdx], d_z[nbIdx]};

                        float distSq = lengthSquared(p - np);

                        if (distSq < r_squared && (INCLUDE_ZERO_DIST || distSq != 0.0f))
                        {
                            if (numFound >= MAX_NEIGHBORS)
                                return numFound;
                            // assert(numFound < MAX_NEIGHBORS);

                            out[numFound] = nbIdx;
                            ++numFound;
                        }
                    }
                }
            }
        }
        return numFound;
    }

    template <bool INCLUDE_ZERO_DIST = true, typename TPartSize, typename Op>
    FUNC void forEachNeighbor(cut::Vec3f p, TPartSize num_particles, const float *d_x, const float *d_y,
                              const float *d_z, Op &op, float h_sq) const
    {
        cut::Vec3i cell = getCell(p);

        for (int z = cut::max(cell.z - 1, 0); z < cut::min(cell.z + 2, (int)size.z); ++z)
        {
            for (int y = cut::max(cell.y - 1, 0); y < cut::min(cell.y + 2, (int)size.y); ++y)
            {
                for (int x = cut::max(cell.x - 1, 0); x < cut::min(cell.x + 2, (int)size.x); ++x)
                {
                    auto nbCellIdx = getIndexFromCell(cut::Vec3i{x, y, z});
                    auto nbCellPtr = cells[nbCellIdx];

                    if (nbCellPtr == EMPTY_CELL)
                        continue;

                    // Find out how many particles are in the current cell
                    auto nIt = nbCellIdx + 1;
                    while (nIt < total_size && cells[nIt] == EMPTY_CELL)
                        ++nIt;

                    int partsInCell;
                    if (nIt == total_size)
                    {
                        partsInCell = num_particles - nbCellPtr;
                    }
                    else
                    {
                        assert(cells[nIt] != EMPTY_CELL);
                        partsInCell = cells[nIt] - nbCellPtr;
                    }

                    assert(partsInCell > 0);

                    // For every particle in the cell, check if it is inside the radius
                    for (int n = 0; n < partsInCell; ++n)
                    {
                        auto nbIdx = nbCellPtr + n;

                        assert(nbIdx < num_particles);

                        cut::Vec3f np{d_x[nbIdx], d_y[nbIdx], d_z[nbIdx]};

                        float distSq = lengthSquared(p - np);

                        if (distSq < h_sq && (INCLUDE_ZERO_DIST || distSq != 0.0f))
                        {
                            op(nbIdx, np);
                        }
                    }
                }
            }
        }
    }

    template <bool INCLUDE_ZERO_DIST = true, typename TPartSize, typename Op>
    FUNC void forEachNeighbor(cut::Vec3f p, TPartSize num_particles, const float *d_x, const float *d_y,
                              const float *d_z, Op &op) const
    {
        forEachNeighbor<INCLUDE_ZERO_DIST>(p, num_particles, d_x, d_y, d_z, op, r_squared);
    }

    template <bool INCLUDE_ZERO_DIST = true, typename TPartSize, typename Op>
    FUNC void forEachNeighbor(cut::Vec3f p, TPartSize num_particles, const cut::Vec3f *d_points, Op &op, float h_sq,
                              int num_neighbors) const
    {
        cut::Vec3i cell = getCell(p);

        for (int z = cut::max(cell.z - num_neighbors, 0); z < cut::min(cell.z + 1 + num_neighbors, (int)size.z); ++z)
        {
            for (int y = cut::max(cell.y - num_neighbors, 0); y < cut::min(cell.y + 1 + num_neighbors, (int)size.y);
                 ++y)
            {
                for (int x = cut::max(cell.x - num_neighbors, 0); x < cut::min(cell.x + 1 + num_neighbors, (int)size.x);
                     ++x)
                {
                    auto nbCellIdx = getIndexFromCell(cut::Vec3i{x, y, z});
                    auto nbCellPtr = cells[nbCellIdx];

                    if (nbCellPtr == EMPTY_CELL)
                        continue;

                    // Find out how many particles are in the current cell
                    auto nIt = nbCellIdx + 1;
                    while (nIt < total_size && cells[nIt] == EMPTY_CELL)
                        ++nIt;

                    int partsInCell;
                    if (nIt == total_size)
                    {
                        partsInCell = num_particles - nbCellPtr;
                    }
                    else
                    {
                        assert(cells[nIt] != EMPTY_CELL);
                        partsInCell = cells[nIt] - nbCellPtr;
                    }

                    assert(partsInCell > 0);

                    // For every particle in the cell, check if it is inside the radius
                    for (int n = 0; n < partsInCell; ++n)
                    {
                        auto nbIdx = nbCellPtr + n;

                        assert(nbIdx < num_particles);

                        cut::Vec3f np = d_points[nbIdx];

                        float distSq = lengthSquared(p - np);

                        if (distSq < h_sq && (INCLUDE_ZERO_DIST || distSq != 0.0f))
                        {
                            op(nbIdx, np);
                        }
                    }
                }
            }
        }
    }
};

//---------------------------------------------------------------------------------------------------------------

template <bool USE_GPU> class UniformGrid
{
public:
    static constexpr int TileSizeX = 4;
    static constexpr int TileSizeY = 4;
    static constexpr int TileSizeZ = 4;

    inline UniformGrid(cut::Vec3u grids, cut::Vec3f pMin, float cellSize)
    {
        size.x = getNextMultiple(grids.x, TileSizeX);
        size.y = getNextMultiple(grids.y, TileSizeY);
        size.z = (grids.z > 1) ? getNextMultiple(grids.z, TileSizeZ) : grids.z;
        cell_size = cellSize;

        cut::Vec3i tileSize{TileSizeX, TileSizeY, TileSizeZ};
        if (is2Dim())
            tileSize.z = 1;

        d_grid.tile_size = tileSize;
        d_grid.size = size;
        d_grid.min_pt = pMin;
        d_grid.inv_cell_size = 1.0 / cell_size;
        d_grid.total_size =
            static_cast<CellIndex>(size.x) * static_cast<CellIndex>(size.y) * static_cast<CellIndex>(size.z);
        d_grid.r_squared = cellSize * cellSize;
    }

    inline UniformGrid(const UniformGrid<false> &grid)
        : size(grid.size)
        , cell_size(grid.cell_size)
        , d_grid(grid.d_grid)
    {
        d_grid.cells = nullptr;
        allocCells();
#ifdef CUDA_SUPPORT
        if (USE_GPU)
            CHECK_CUDA(cudaMemcpy(d_grid.cells, grid.getCells(), sizeof(GridCell) * size.x * size.y * size.z,
                                  cudaMemcpyHostToDevice));
        else
#endif
            std::copy(grid.getCells(), grid.getCells() + size.x * size.y * size.z, d_grid.cells);
    }

#ifdef CUDA_SUPPORT
    inline UniformGrid(const UniformGrid<true> &grid)
        : size(grid.size)
        , cell_size(grid.cell_size)
        , d_grid(grid.d_grid)
    {
        d_grid.cells = nullptr;
        allocCells();
        if (USE_GPU)
            CHECK_CUDA(cudaMemcpy(d_grid.cells, grid.getCells(), sizeof(GridCell) * size.x * size.y * size.z,
                                  cudaMemcpyDeviceToDevice));
        else
            CHECK_CUDA(cudaMemcpy(d_grid.cells, grid.getCells(), sizeof(GridCell) * size.x * size.y * size.z,
                                  cudaMemcpyDeviceToHost));
    }
#endif

    inline ~UniformGrid()
    {
#ifdef CUDA_SUPPORT
        if (USE_GPU)
            cut::safeFreeUnified(d_grid.cells);
        else
#endif
            free(d_grid.cells);
    }

    inline void allocCells()
    {
        assert(size.x <= 8192 && size.y <= 8192 && size.z <= 8192);
#ifdef CUDA_SUPPORT
        if (USE_GPU)
            d_grid.cells = (GridCell *)UNIFIED_MALLOC(sizeof(GridCell) * size.x * size.y * size.z);
        else
#endif
            d_grid.cells = (GridCell *)malloc(sizeof(GridCell) * size.x * size.y * size.z);
    }

    inline bool is2Dim() const { return size.z == 1; }

    GridCell *getCells() const { return d_grid.cells; }
    cut::Vec3u getSize() const { return size; }
    uint32_t getTotalSize() const { return d_grid.total_size; }

    cut::Vec3f getGridMin() const { return d_grid.min_pt; }
    cut::Vec3f getGridMax() const { return d_grid.min_pt + cell_size * cut::make_vec3f(size); }

    inline const UniformGridDevice &getOnDevice() const { return d_grid; }

    cut::Vec3u size;
    float cell_size;

    UniformGridDevice d_grid;
}; // namespace grid

using UniformGridGPU = UniformGrid<true>;
using UniformGridCPU = UniformGrid<false>;

extern void setGridCellsCPU(GridCell *cells, const CellIndex *cellPtr, uint32_t num_particles);
extern void setGridCellsGPU(GridCell *cells, const CellIndex *cellPtr, uint32_t num_particles);

extern unique_ptr<UniformGridGPU> createGridFromDomainGPU(const float *x, const float *y, const float *z,
                                                          uint32_t num_particles, float r);

extern unique_ptr<UniformGridCPU> createGridFromDomainCPU(const float *x, const float *y, const float *z,
                                                          uint32_t num_particles, float r);

template <bool USE_GPU>
inline unique_ptr<UniformGrid<USE_GPU>> createGridFromDomain(float r, cut::Vec3f min, cut::Vec3f max)
{
    // Calculate the grid size
    float cellSize = r;
    auto grids = (max - min) / cellSize;

    return std::move(unique_ptr<UniformGrid<USE_GPU>>(
        new UniformGrid<USE_GPU>{cut::Vec3u{static_cast<uint32_t>(grids.x + 1), static_cast<uint32_t>(grids.y + 1),
                                            static_cast<uint32_t>(grids.z + 1)},
                                 min, cellSize}));
}

#endif // PARTICLE_GRID_CUH