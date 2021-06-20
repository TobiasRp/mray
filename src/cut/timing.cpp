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
#include "timing.h"
#include "timing_log_writer.h"

#include <iostream>

using namespace std;

TimeLogger g_TimeLogger;

void TimeLogger::release() {}

void TimeLogger::startCUDA(const string &name)
{
#ifdef CUDA_SUPPORT
    auto it = m_cuda_events.find(name);
    if (it != std::end(m_cuda_events))
    {
        cudaEventRecord(it->second.start);
    }
    else
    {
        CUDAEventPair pair;
        cudaEventCreate(&pair.start);
        cudaEventCreate(&pair.stop);
        cudaEventRecord(pair.start);

        m_cuda_events[name] = pair;
    }
#else
    UNUSED(name);
#endif // #ifdef CUDA_SUPPORT
}

void TimeLogger::endCUDA(const string &name)
{
#ifdef CUDA_SUPPORT
    auto it = m_cuda_events.find(name);
    if (it != std::end(m_cuda_events))
        cudaEventRecord(it->second.stop);
    else
        assert(false);
#else
    UNUSED(name);
#endif // #ifdef CUDA_SUPPORT
}

void TimeLogger::startCPU(const std::string &name)
{
    m_timepoints[name].start = chrono::high_resolution_clock::now();
}

void TimeLogger::endCPU(const std::string &name)
{
    auto end = chrono::high_resolution_clock::now();
    auto &tp = m_timepoints[name];
    tp.total_ns += chrono::duration_cast<chrono::nanoseconds>(end - tp.start).count();
}

inline float toMS(size_t x)
{
    return x / 1000000.0f;
}

void TimeLogger::syncAndReport()
{
#ifdef CUDA_SUPPORT
    for (auto pair : m_cuda_events)
    {
        cudaEventSynchronize(pair.second.stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, pair.second.start, pair.second.stop);
        TimeLogWriter::logCUDAEvent(milliseconds, pair.first);
        std::cout << pair.first.c_str() << "(CUDA): " << milliseconds << "ms" << std::endl;
    }
    m_cuda_events.clear();
#endif // CUDA_SUPPORT

    for (auto point : m_timepoints)
    {
        TimeLogWriter::logCPUEvent(
            std::chrono::time_point_cast<std::chrono::milliseconds>(point.second.start).time_since_epoch().count(),
            toMS(point.second.total_ns), point.first);
        std::cout << point.first.c_str() << "(CPU): " << point.second.total_ns / (1000 * 1000) << "ms\n";
    }
    m_timepoints.clear();
    std::cout.flush();
}
