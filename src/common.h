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
#ifndef COMMON_DATA_H
#define COMMON_DATA_H

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif

// To avoid compiler warnings for unused variables/parameters
#define UNUSED(x) (void)(x)

#include <cmath>

#ifdef _WIN32
#include <windows.h>
#undef min
#undef max
#endif

#include <cassert>
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

using std::array;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::unordered_map;
using std::vector;

using Byte = unsigned char;

#ifdef CUDA_SUPPORT
#include "cut/cuda_common.h"
#endif // CUDA_SUPPORT

#include "cut/cuda_math.h"
using cut::Vec2f;
using cut::Vec3f;
using cut::Vec3i;
using cut::Vec4f;

/**
 * @brief Exception to be used if the data is _invalid_.
 */
class invalid_data : std::exception
{
    const char *m_msg;

public:
    invalid_data(const char *msg)
        : m_msg(msg)
    {
    }
    virtual const char *what() const noexcept override { return m_msg; }
};

/**
 * @brief Exception to be used if arguments are _invalid_.
 */
class invalid_args : std::exception
{
    const char *m_msg;

public:
    invalid_args(const char *msg)
        : m_msg(msg)
    {
    }
    virtual const char *what() const noexcept override { return m_msg; }
};

/** Returns true if the given filename refers to an existing file */
extern bool isFile(const char *filename);

inline string to_lower(string str)
{
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    return str;
}

inline string to_upper(string str)
{
    std::transform(str.begin(), str.end(), str.begin(), ::toupper);
    return str;
}

/**
 * @brief A range consists of a minimum and a maximum value.
 */
template <typename T> struct Range
{
    FUNC Range() {}

    FUNC Range(T min, T max)
        : min(min)
        , max(max)
    {
    }

    T min;
    T max;
};

#define MAX_UINT16 65535.f

#define INVALID_DENSITY 1e+33f

struct DensityBound
{
    uint16_t lower;
    uint16_t upper;

    DensityBound() = default;
    FUNC DensityBound(float l, float u) { set(l, u); }

    FUNC void set(float l, float u)
    {
        lower = static_cast<uint16_t>(l * MAX_UINT16);
        upper = static_cast<uint16_t>(u * MAX_UINT16);
    }

    FUNC Vec2f get() const { return Vec2f(lower / (MAX_UINT16), upper / (MAX_UINT16)); }
};

enum ErrorType
{
    MAX_ERROR = 0,
    RMSE = 1,
    ERROR_PERCENTILE = 2
};

#endif // COMMON_DATA_H
