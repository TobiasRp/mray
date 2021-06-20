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
#include "particle_loader.h"

#include "H5Cpp.h"
#include <map>
#include <algorithm>
#include <fstream>

#include "cut/extrema.h"

#ifdef _WIN32
#include <cctype> // std::isdigit
#endif

inline int getIndexFromGroup(const string &name, int idx)
{
    try
    {
        // Try to parse the time step index from the name

        auto numBegin = name.begin();
        while (numBegin != name.end() && !std::isdigit(*numBegin))
            ++numBegin;

        auto numEnd = numBegin;
        while (numEnd != name.end() && std::isdigit(*numEnd))
            ++numEnd;

        string number(numBegin, numEnd);

        return std::stoi(number);
    }
    catch (std::exception &exc)
    {
        return idx;
    }
}

inline vector<uint32_t> createIndexMap(const H5::H5File *file)
{
    auto numSteps = file->getNumObjs();

    std::map<uint32_t, uint32_t> map;

    for (size_t i = 0; i < numSteps; ++i)
    {
        string stepName = file->getObjnameByIdx(i);

        auto substr = stepName.substr(0, 4);
        if (substr != "Step")
            continue;

        if (file->getObjTypeByIdx(i) != H5G_GROUP)
            continue;

        auto idx = getIndexFromGroup(stepName, i);

        map[idx] = i;
    }

    vector<uint32_t> values;
    values.reserve(map.size());

    for (auto it = map.begin(); it != map.end(); ++it)
        values.push_back(it->second);

    return values;
}

ParticleLoader::ParticleLoader(const std::string &filename)
{
    m_file = std::make_unique<H5::H5File>(filename.c_str(), H5F_ACC_RDONLY);
    m_file_map = createIndexMap(m_file.get());
    m_num_timesteps = m_file_map.size();

    if (m_num_timesteps == 0)
        throw invalid_data("The stream doesn't contain any time steps");
}

ParticleLoader::~ParticleLoader() {}

inline float readTime(H5::Group &step, int stepNr)
{
    float timeValue;
    if (step.attrExists("TimeValue"))
    {
        auto time = step.openAttribute("TimeValue");

        if (time.getDataType().getClass() == H5T_FLOAT && time.getDataType().getSize() == 4)
            time.read(time.getDataType(), &timeValue);
        else if (time.getDataType().getClass() == H5T_FLOAT && time.getDataType().getSize() == 8)
        {
            double dtime;
            time.read(time.getDataType(), &dtime);
            timeValue = static_cast<float>(dtime);
        }
        else if (time.getDataType().getClass() == H5T_STRING)
        {
            auto size = time.getStorageSize();
            vector<char> timeStr(size + 1);
            time.read(time.getDataType(), &timeStr[0]);

            timeStr.back() = '\0';
            timeValue = std::strtof(&timeStr[0], nullptr);
        }
        else
            assert(false); // Either the data is incorrect >or< we have to add another type here!
    }
    else
    {
        timeValue = static_cast<float>(stepNr);
    }
    return timeValue;
}

inline size_t readNumValues(H5::Group &step)
{
    size_t numValues;

    if (step.attrExists("num"))
    {
        uint32_t num;
        auto numAttr = step.openAttribute("num");

        assert(numAttr.getDataType().getClass() == H5T_INTEGER && numAttr.getDataType().getSize() == 4);
        numAttr.read(numAttr.getDataType(), &num);
        numValues = num;
    }
    else
    {
        int datasetIdx = 0;
        while (step.getObjTypeByIdx(datasetIdx) != H5G_DATASET)
            ++datasetIdx;

        auto dataset = step.openDataSet(step.getObjnameByIdx(datasetIdx));
        numValues = dataset.getSpace().getSimpleExtentNpoints();
    }
    return numValues;
}

bool ParticleLoader::getProperty(std::string name, float &result) const
{
    if (!m_file->attrExists(name))
        return false;

    auto attr = m_file->openAttribute(name);
    attr.read(attr.getDataType(), &result);
    return true;
}

inline bool hdf5_name_exists(hid_t id, const std::string &path)
{
    return H5Lexists( id, path.c_str(), H5P_DEFAULT ) > 0;
}

void ParticleLoader::fetch(int t, string var, vector<float> &out) const
{
    assert(m_file);

    auto fileIdx = m_file_map[t];

    H5::DataSet set;
    try
    {
        auto group = m_file->openGroup(m_file->getObjnameByIdx(fileIdx));

        if (!hdf5_name_exists(group.getLocId(), var.c_str()))
        {
            // Try uppercase...
            var = to_upper(var);
            if (!hdf5_name_exists(group.getLocId(), var.c_str()))
                throw invalid_data("Variable doesn't exist!");
        }

        set = group.openDataSet(var);
    }
    catch (H5::FileIException &e)
    {
    }

    auto byte_size = set.getInMemDataSize();
    auto num_parts = byte_size / sizeof(float);

    out.resize(num_parts);
    set.read(out.data(), set.getDataType());
}

Particles ParticleLoader::get(int t, string var, float smoothing_length) const
{
    vector<float> x, y, z, values;

    fetch(t, "x", x);
    fetch(t, "y", y);
    fetch(t, "z", z);
    fetch(t, var, values);

    Range<Vec3f> domain;
    cut::find_minmax(x.data(), x.size(), domain.min.x, domain.max.x);
    cut::find_minmax(y.data(), y.size(), domain.min.y, domain.max.y);
    cut::find_minmax(z.data(), z.size(), domain.min.z, domain.max.z);

    Particles part(domain, x, y, z, values, smoothing_length);
    return part;
}