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
#ifndef TIMER_WRITER_H
#define TIMER_WRITER_H

#include "H5Cpp.h"
#include <string>
#include <memory>

/**
 * @brief This class provides a basic interface to write the timing logs to a hdf5 file.
 *
 * @detail This class is a singleton. A single instance holds the access to a hdf5 file.
 * This class provides functionality to log events to a hdf5 file that contains multiple groups
 * for different event sources.
 **/
class TimeLogWriter
{
    struct LogEntry
    {
        float timestamp;
        float runtime;
        const char *eventName;
    };

public:
    static void init(std::string file, std::string grp);
    static void init_no_logging();

    static void destroy() { m_instance = nullptr; }

    static void logCPUEvent(float timestamp, float runtime, std::string eventName);
#ifdef CUDA_SUPPORT
    static void logCUDAEvent(float runtime, std::string eventName);
#endif

    static TimeLogWriter *GetInstance();

protected:
    TimeLogWriter();
    TimeLogWriter(std::string file, std::string grp);
    ~TimeLogWriter();

    void setUpFile(std::string file, std::string grp);
    void closeFile();

    bool m_write_logs = false;
    bool m_is_file_init = false;

    hsize_t m_current_dim_cpu[1] = {1};
    hsize_t m_current_dim_gpu[1] = {1};
    hsize_t m_max_dim[1] = {H5S_UNLIMITED};
    hsize_t m_chunk_dim[1] = {5};

    H5::CompType m_log_entry_type;
    std::unique_ptr<H5::H5File> m_log_file;

    H5::Group m_cpu_group;
    std::unique_ptr<H5::DataSet> m_cpu_data_set;

#ifdef CUDA_SUPPORT
    hsize_t m_current_dim_cuda[1] = {1};

    H5::Group m_cuda_group;
    std::unique_ptr<H5::DataSet> m_cuda_data_set;
#endif

private:
    static TimeLogWriter *m_instance;
};

#endif
