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
#include "timing_log_writer.h"

#include "H5Cpp.h"
#include <iostream>
#include <chrono>
#include "../common.h"

TimeLogWriter *TimeLogWriter::m_instance = NULL;

inline bool hdf5_name_exists(hid_t id, const std::string &path)
{
    return H5Lexists( id, path.c_str(), H5P_DEFAULT ) > 0;
}

void TimeLogWriter::logCPUEvent(float timestamp, float runtime, const std::string eventName)
{
    auto instance = GetInstance();
    if (!instance->m_write_logs || !instance->m_is_file_init)
        return;

    LogEntry new_entry;
    new_entry.timestamp = timestamp;
    new_entry.runtime = runtime;
    new_entry.eventName = eventName.c_str();
    hsize_t entry_dim[1] = {1};

    hsize_t offset[1] = {instance->m_current_dim_cpu[0]};
    instance->m_current_dim_cpu[0]++;
    instance->m_cpu_data_set->extend(instance->m_current_dim_cpu);

    auto file_space = new H5::DataSpace(instance->m_cpu_data_set->getSpace());
    file_space->selectHyperslab(H5S_SELECT_SET, entry_dim, offset);

    auto mem_space = new H5::DataSpace(1, entry_dim, NULL);

    try
    {
        H5::Exception::dontPrint();
        instance->m_cpu_data_set->write(&new_entry, instance->m_log_entry_type, *mem_space, *file_space);
    }
    catch (H5::DataSetIException &error)
    {
        std::cout << error.getDetailMsg();
    }
    delete file_space;
    delete mem_space;
}

#ifdef CUDA_SUPPORT
void TimeLogWriter::logCUDAEvent(float runtime, const std::string eventName)
{
    auto instance = GetInstance();
    if (!instance->m_write_logs)
        return;

    LogEntry new_entry;
    new_entry.timestamp = instance->m_current_dim_cuda[0];
    new_entry.runtime = runtime;
    new_entry.eventName = eventName.c_str();
    hsize_t entry_dim[1] = {1};

    hsize_t offset[1] = {instance->m_current_dim_cuda[0]};
    instance->m_current_dim_cuda[0]++;
    instance->m_cuda_data_set->extend(instance->m_current_dim_cuda);

    auto file_space = new H5::DataSpace(instance->m_cuda_data_set->getSpace());
    file_space->selectHyperslab(H5S_SELECT_SET, entry_dim, offset);

    auto mem_space = new H5::DataSpace(1, entry_dim, NULL);

    try
    {
        H5::Exception::dontPrint();
        instance->m_cuda_data_set->write(&new_entry, instance->m_log_entry_type, *mem_space, *file_space);
    }
    catch (H5::DataSetIException &error)
    {
        std::cout << error.getDetailMsg();
    }
    delete file_space;
    delete mem_space;
}
#endif

TimeLogWriter *TimeLogWriter::GetInstance()
{
    assert(m_instance);
    return TimeLogWriter::m_instance;
}

TimeLogWriter::TimeLogWriter(string file, string grp)
    : m_write_logs(true)
{
    setUpFile(file, grp);
}

TimeLogWriter::TimeLogWriter()
    : m_write_logs(false)
{
}

void TimeLogWriter::init(string file, string grp)
{
    TimeLogWriter::m_instance = new TimeLogWriter(file, grp);
}

void TimeLogWriter::init_no_logging()
{
    TimeLogWriter::m_instance = new TimeLogWriter();
}

TimeLogWriter::~TimeLogWriter()
{
    closeFile();
}

void TimeLogWriter::setUpFile(string file, string grp)
{
    try
    {
        //H5::Exception::dontPrint();

        LogEntry initLogEntry;
        initLogEntry.eventName = "InitLog";
        initLogEntry.timestamp = 0.0f;
        initLogEntry.runtime = 0.0f;

        //        auto startTimePoint = std::chrono::system_clock::now();
        //        time_t startTime = std::chrono::system_clock::to_time_t(startTimePoint);
        //        auto localTime = localtime(&startTime);

        if (isFile(file.c_str()))
            m_log_file = std::make_unique<H5::H5File>(file, H5F_ACC_RDWR);
        else
            m_log_file = std::make_unique<H5::H5File>(file, H5F_ACC_TRUNC);

        std::cout << "Writing logs to: " << file << std::endl;

        m_log_entry_type = H5::CompType(sizeof(LogEntry));
        m_log_entry_type.insertMember("Timestamp", HOFFSET(LogEntry, timestamp), H5::PredType::NATIVE_FLOAT);
        m_log_entry_type.insertMember("Runtime (ms)", HOFFSET(LogEntry, runtime), H5::PredType::NATIVE_FLOAT);

        H5::StrType stype(H5::PredType::C_S1, H5T_VARIABLE);
        m_log_entry_type.insertMember("EventName", HOFFSET(LogEntry, eventName), stype);

        H5::DataSpace data_space(1, m_current_dim_cpu, m_max_dim);
        H5::DSetCreatPropList properties;
        properties.setChunk(1, m_chunk_dim);

        int i = 0;
        auto grp_name = grp;
        while(hdf5_name_exists(m_log_file->getLocId(), grp_name))
        {
            grp_name = grp + std::to_string(i);
            ++i;
        }

        auto hgroup = m_log_file->createGroup(grp_name);

        m_cpu_group = m_log_file->createGroup(grp_name + string("/CPU"));
        m_cpu_data_set =
            std::make_unique<H5::DataSet>(m_cpu_group.createDataSet("Log", m_log_entry_type, data_space, properties));
        m_cpu_data_set->write(&initLogEntry, m_log_entry_type);

#ifdef CUDA_SUPPORT
        m_cuda_group = m_log_file->createGroup(grp_name + string("/CUDA"));
        m_cuda_data_set =
            std::make_unique<H5::DataSet>(m_cuda_group.createDataSet("Log", m_log_entry_type, data_space, properties));
        m_cuda_data_set->write(&initLogEntry, m_log_entry_type);
#endif
    }
    catch (H5::FileIException &error)
    {
        std::cout << error.getDetailMsg();
        return;
    }

    m_is_file_init = true;
}

void TimeLogWriter::closeFile()
{
    m_log_entry_type.close();

    m_cpu_group.close();
#ifdef CUDA_SUPPORT
    m_cuda_group.close();
#endif

    m_is_file_init = false;
}
