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
#ifndef DEVICE_TIMER_H
#define DEVICE_TIMER_H

#include "../common.h"

#include <chrono>
#include <unordered_map>

#ifdef CUDA_SUPPORT
#ifdef USE_NVTX
#include "nvToolsExt.h"
class Tracer {
public:
    Tracer(const char* name) {
        nvtxRangePushA(name);
    }
    ~Tracer() {
        nvtxRangePop();
    }
};
#define NVTX_RANGE(name) Tracer uniq_name_using_macros(name);
#else
#define NVTX_RANGE(name)
#endif
#endif

/**
 * @brief The TimeLogger class is useful for detailed timings and statistics,
 * but its impact on performance is probably too much for production use.
 */
class TimeLogger
{
    struct TimePointPair
    {
        TimePointPair()
            : total_ns(0)
        {
        }

        std::chrono::high_resolution_clock::time_point start;
        uint64_t total_ns;
    };

#ifdef CUDA_SUPPORT
    struct CUDAEventPair
    {
        cudaEvent_t start;
        cudaEvent_t stop;
    };
#endif

public:
    void release();

    void startCUDA(const string &name);

    void endCUDA(const string &name);

    void startCPU(const string &name);

    void endCPU(const string &name);

    /**
     * Synchronizes the CPU with the GPU and prints all statistics
     */
    void syncAndReport();

private:
    std::unordered_map<string, TimePointPair> m_timepoints;

#ifdef CUDA_SUPPORT
    std::unordered_map<std::string, CUDAEventPair> m_cuda_events;
#endif
};

extern TimeLogger g_TimeLogger;

class ScopedCUDAQuery
{
public:
    inline ScopedCUDAQuery(const string &name)
        : m_name(name)
    {
        g_TimeLogger.startCUDA(name);
    }
    inline ~ScopedCUDAQuery() { g_TimeLogger.endCUDA(m_name); }

private:
    string m_name;
};

class ScopedCPUQuery
{
public:
    inline ScopedCPUQuery(const string &name)
        : m_name(name)
    {
        g_TimeLogger.startCPU(name);
    }
    inline ~ScopedCPUQuery() { g_TimeLogger.endCPU(m_name); }

private:
    string m_name;
};

#define MERGE_(A, B) A##B

#define START_CPU_QUERY(name) g_TimeLogger.startCPU(name);
#define END_CPU_QUERY(name) g_TimeLogger.endCPU(name);
#define SCOPED_CPU_QUERY(name) ScopedCPUQuery MERGE_(queryCPU_, __LINE__)(name);
#define START_CUDA_QUERY(name) g_TimeLogger.startCUDA(name);
#define END_CUDA_QUERY(name) g_TimeLogger.endCUDA(name);
#define SCOPED_CUDA_QUERY(name) ScopedCUDAQuery MERGE_(queryCUDA_, __LINE__)(name); \
                                NVTX_RANGE(name)

#endif // DEVICE_TIMER_H
