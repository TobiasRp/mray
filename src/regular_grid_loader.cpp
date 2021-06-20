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
#include "regular_grid_loader.h"
#include <netcdf.h>
#include <iostream>

using namespace cut;

inline void check_nc(int e)
{
    if (e != NC_NOERR)
    {
        std::cerr << nc_strerror(e) << std::endl;
        throw invalid_data(nc_strerror(e));
    }
}
#define CHECK_NC(e) check_nc(e)

RegularGridLoader::RegularGridLoader(string file)
{
    CHECK_NC(nc_open(file.c_str(), NC_NOWRITE, &m_fid));

    // Read dimensions
    int xdim_id, ydim_id, zdim_id, tdim_id;
    cut::Vec4<size_t> dims;
    CHECK_NC(nc_inq_dimid(m_fid, "xdim", &xdim_id));
    CHECK_NC(nc_inq_dimid(m_fid, "ydim", &ydim_id));
    CHECK_NC(nc_inq_dimid(m_fid, "zdim", &zdim_id));

    CHECK_NC(nc_inq_dimlen(m_fid, xdim_id, &dims.x));
    CHECK_NC(nc_inq_dimlen(m_fid, ydim_id, &dims.y));
    CHECK_NC(nc_inq_dimlen(m_fid, zdim_id, &dims.z));

    if (nc_inq_dimid(m_fid, "tdim", &tdim_id) != NC_NOERR)
        dims.w = 1;
    else
        CHECK_NC(nc_inq_dimlen(m_fid, tdim_id, &dims.w));

    m_dim = Vec4i(dims.x, dims.y, dims.z, dims.w);

    // Read spatial and temporal min/max
    CHECK_NC(nc_get_att_float(m_fid, NC_GLOBAL, "spatial_min", &m_space.min.x));
    CHECK_NC(nc_get_att_float(m_fid, NC_GLOBAL, "spatial_max", &m_space.max.x));
    CHECK_NC(nc_get_att_float(m_fid, NC_GLOBAL, "time_domain", &m_time.min));
}

RegularGridLoader::~RegularGridLoader()
{
    CHECK_NC(nc_close(m_fid));
}

RegularGrid RegularGridLoader::get(int t, string var) const
{
    int id;
    CHECK_NC(nc_inq_varid(m_fid, var.c_str(), &id));

    nc_type vartype;
    CHECK_NC(nc_inq_vartype(m_fid, id, &vartype));
    RegularGrid::DataType dtype;

    vector<float> data;
    if (vartype == NC_UBYTE)
    {
        dtype = RegularGrid::DataType::U8;
        size_t data_size =
            static_cast<size_t>(m_dim.x) * static_cast<size_t>(m_dim.y) * static_cast<size_t>(m_dim.z) / sizeof(float);
        data.resize(data_size);
    }
    else if (vartype == NC_FLOAT)
    {
        dtype = RegularGrid::DataType::F32;
        data.resize(m_dim.x * m_dim.y * m_dim.z);
    }
    else
    {
        dtype = RegularGrid::DataType::F32;
        assert(false);
    }

    if (m_dim.w > 1)
    {
        size_t start[] = {static_cast<size_t>(t), 0, 0, 0};
        size_t count[] = {1, static_cast<size_t>(m_dim.z), static_cast<size_t>(m_dim.y), static_cast<size_t>(m_dim.x)};

        if (dtype == RegularGrid::DataType::U8)
            CHECK_NC(nc_get_vara_ubyte(m_fid, id, start, count, reinterpret_cast<uint8_t *>(data.data())));
        else if (dtype == RegularGrid::DataType::F32)
            CHECK_NC(nc_get_vara_float(m_fid, id, start, count, data.data()));
    }
//    else if (m_dim.z > 512)
//    {
//        assert(t == 0);
//
//        size_t block_size = 1;
//        size_t num_blocks = m_dim.z / block_size;
//        for (int z = 0; z < num_blocks; ++z)
//        {
//            size_t z_offset = z * block_size;
//            size_t z_size = cut::min(block_size, m_dim.z - z_offset);
//
//            size_t start[] = {static_cast<size_t>(t), static_cast<size_t>(z_offset), 0, 0};
//            size_t count[] = {1, static_cast<size_t>(z_size), static_cast<size_t>(m_dim.y),
//                              static_cast<size_t>(m_dim.x)};
//
//            size_t data_offset = z_offset * static_cast<size_t>(m_dim.x * m_dim.y);
//
//            if (dtype == RegularGrid::DataType::U8)
//                CHECK_NC(nc_get_vara_ubyte(m_fid, id, start, count,
//                                           reinterpret_cast<uint8_t *>(&data[data_offset / sizeof(float)])));
//            else if (dtype == RegularGrid::DataType::F32)
//                CHECK_NC(nc_get_vara_float(m_fid, id, start, count, &data[data_offset]));
//        }
//    }
    else
    {
        assert(t == 0);

        if (dtype == RegularGrid::DataType::U8)
            CHECK_NC(nc_get_var_ubyte(m_fid, id, reinterpret_cast<uint8_t *>(data.data())));
        else if (dtype == RegularGrid::DataType::F32)
            CHECK_NC(nc_get_var_float(m_fid, id, data.data()));
    }

    return RegularGrid(Vec3i(m_dim.x, m_dim.y, m_dim.z), m_space, std::move(data), dtype);
}