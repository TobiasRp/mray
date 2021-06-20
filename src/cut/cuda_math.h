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
#ifndef CUDA_MATH
#define CUDA_MATH

#ifdef _MSC_VER
#pragma warning(disable : 4522) /* Ignore warnings about multiple assignment operators */
#endif

#include "cuda_common.h"
#include <cassert>
#include <cmath>

#ifdef _WIN32
#undef min
#undef max
#endif

namespace cut
{

template <typename T> FUNC void swap(T &a, T &b)
{
    T temp = a;
    a = b;
    b = temp;
}

template <typename T> FUNC void orderDescending(T &a, T &b)
{
    if (b > a)
        swap(a, b);
}

template <typename T> FUNC void orderDescending(T &a, T &b, T &c)
{
    if (b >= a && b >= c)
    {
        swap(a, b);
    }
    else if (c >= a && c >= b)
    {
        swap(a, c);
    }

    if (b < c)
        swap(b, c);
}

template <typename VECTOR> FUNC VECTOR lerp(const VECTOR &a, const VECTOR &b, float t)
{
    return (1 - t) * a + t * b;
}

template <typename T> FUNC const T min(T x, T y)
{
    return (x < y) ? x : y;
}

template <typename T> FUNC const T max(T x, T y)
{
    return (x > y) ? x : y;
}

template <typename T> FUNC const T abs(T x)
{
    return (x < 0) ? -x : x;
}

template <typename T> class Vec2
{
public:
    Vec2() = default;

    FUNC Vec2(T p1, T p2)
        : x(p1)
        , y(p2)
    {
    }

    FUNC Vec2(const T *arr)
        : x(arr[0])
        , y(arr[1])
    {
    }

    FUNC explicit Vec2(T scalar)
        : x(scalar)
        , y(scalar)
    {
    }

#ifdef CUDA_SUPPORT
    FUNC explicit Vec2(float4 vec)
        : x(vec.x)
        , y(vec.y)
    {
    }
#endif

#ifdef __CUDACC__
    FUNC Vec2<T> &operator=(const Vec2 &v)
    {
        x = v.x;
        y = v.y;
        return *this;
    }

    FUNC volatile Vec2<T> &operator=(volatile const Vec2 &v) volatile
    {
        x = v.x;
        y = v.y;
        return *this;
    }
#endif

    FUNC const T &operator[](const size_t i) const
    {
        assert(i < 2);
        return *(&x + i);
    }

    FUNC T &operator[](const size_t i)
    {
        assert(i < 2);
        return *(&x + i);
    }

    FUNC bool operator==(Vec2 other) { return x == other.x && y == other.y; }

    FUNC bool operator!=(Vec2 other) { return x != other.x || y != other.y; }

    FUNC Vec2 &operator+=(const Vec2 &other)
    {
        x += other.x;
        y += other.y;
        return *this;
    }

    FUNC Vec2 &operator-=(const Vec2 &other)
    {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    FUNC Vec2 &operator*=(const T scalar)
    {
        x *= scalar;
        y *= scalar;
        return *this;
    }

    FUNC Vec2 &operator/=(const T scalar)
    {
        // assert(scalar != 0);
        T inv = 1 / scalar;
        x *= inv;
        y *= inv;
        return *this;
    }

public:
    T x, y;
};

template <typename T> FUNC Vec2<T> operator+(const Vec2<T> &vec1, const Vec2<T> &vec2)
{
    return Vec2<T>(vec1.x + vec2.x, vec1.y + vec2.y);
}

template <typename T> FUNC Vec2<T> operator-(const Vec2<T> &vec1, const Vec2<T> &vec2)
{
    return Vec2<T>(vec1.x - vec2.x, vec1.y - vec2.y);
}

template <typename T> FUNC Vec2<T> operator-(const Vec2<T> &vec)
{
    return Vec2<T>(-vec.x, -vec.y);
}

template <typename T> FUNC Vec2<T> operator*(const Vec2<T> &vec, T scalar)
{
    return Vec2<T>(vec.x * scalar, vec.y * scalar);
}

template <typename T> FUNC Vec2<T> operator*(T scalar, const Vec2<T> &vec)
{
    return Vec2<T>(vec.x * scalar, vec.y * scalar);
}

template <typename T> FUNC Vec2<T> operator/(const Vec2<T> &vec, T scalar)
{
    // assert(scalar != 0);
    T inv = 1 / scalar;
    return Vec2<T>(vec.x * inv, vec.y * inv);
}

template <typename T> FUNC Vec2<T> mul(const Vec2<T> &vec1, const Vec2<T> &vec2)
{
    return Vec2<T>(vec1.x * vec2.x, vec1.y * vec2.y);
}

template <typename T> FUNC Vec2<T> div(const Vec2<T> &vec1, const Vec2<T> &vec2)
{
    return Vec2<T>(vec1.x / vec2.x, vec1.y / vec2.y);
}

template <typename T> FUNC T dot(const Vec2<T> &vec1, const Vec2<T> &vec2)
{
    return vec1.x * vec2.x + vec1.y * vec2.y;
}

template <typename T> FUNC T lengthSquared(const Vec2<T> &v)
{
    return (v.x * v.x + v.y * v.y);
}

template <typename T> FUNC T length(const Vec2<T> &v)
{
    return std::sqrt(v.x * v.x + v.y * v.y);
}

template <typename T> FUNC Vec2<T> normalize(const Vec2<T> &vec)
{
    T inv = 1 / std::sqrt(vec.x * vec.x + vec.y * vec.y);
    return Vec2<T>(vec.x * inv, vec.y * inv);
}

template <> FUNC Vec2<float> normalize(const Vec2<float> &vec)
{
#ifdef __CUDACC__
    float inv = rsqrtf(vec.x * vec.x + vec.y * vec.y);
#else
    float inv = 1.0f / sqrtf(vec.x * vec.x + vec.y * vec.y);
#endif
    return Vec2<float>(vec.x * inv, vec.y * inv);
}

template <typename T> FUNC Vec2<T> getOrthogonalCCW(const Vec2<T> &v)
{
    return Vec2<T>(v.y, -v.x);
}

template <typename T> FUNC Vec2<T> getOrthogonalCW(const Vec2<T> &v)
{
    return Vec2<T>(-v.y, v.x);
}

using Vec2f = Vec2<float>;
using Vec2d = Vec2<double>;
using Vec2i = Vec2<int>;
using Vec2u = Vec2<uint32_t>;
using Vec2s = Vec2<short>;

template <> FUNC const Vec2i min(const Vec2i a, const Vec2i b)
{
    return Vec2i(min(a.x, b.x), min(a.y, b.y));
}

template <> FUNC const Vec2i max(const Vec2i a, const Vec2i b)
{
    return Vec2i(max(a.x, b.x), max(a.y, b.y));
}

template <> FUNC const Vec2<uint16_t> min(const Vec2<uint16_t> a, const Vec2<uint16_t> b)
{
    return Vec2<uint16_t>(min(a.x, b.x), min(a.y, b.y));
}

template <> FUNC const Vec2<uint16_t> max(const Vec2<uint16_t> a, const Vec2<uint16_t> b)
{
    return Vec2<uint16_t>(max(a.x, b.x), max(a.y, b.y));
}

template <> FUNC const Vec2f min(const Vec2f a, const Vec2f b)
{
    return Vec2f(min(a.x, b.x), min(a.y, b.y));
}

template <> FUNC const Vec2f max(const Vec2f a, const Vec2f b)
{
    return Vec2f(max(a.x, b.x), max(a.y, b.y));
}

template <typename T> FUNC const Vec2<T> abs(Vec2<T> v)
{
    return Vec2<T>(abs(v.x), abs(v.y));
}

// template<typename T>
// FUNC void swap(Vec2<T>& lhs, Vec2<T>& rhs)
//{
//	int temp = lhs.x;
//	lhs.x = rhs.x;
//	rhs.x = temp;
//
//	temp = lhs.y;
//	lhs.y = rhs.y;
//	rhs.y = temp;
//}

template <typename T> class Vec3
{
public:
    Vec3() = default;

    FUNC Vec3(T p1, T p2, T p3)
        : x(p1)
        , y(p2)
        , z(p3)
    {
    }

    FUNC Vec3(Vec2<T> v, T p3)
        : x(v.x)
        , y(v.y)
        , z(p3)
    {
    }

    FUNC Vec3(const T *arr)
        : x(arr[0])
        , y(arr[1])
        , z(arr[2])
    {
    }

    FUNC explicit Vec3(T scalar)
        : x(scalar)
        , y(scalar)
        , z(scalar)
    {
    }

#ifdef CUDA_SUPPORT
    FUNC explicit Vec3(float4 vec)
        : x(vec.x)
        , y(vec.y)
        , z(vec.z)
    {
    }
#endif

#ifdef __CUDACC__
    FUNC Vec3<T> &operator=(const Vec3 &v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }

    FUNC volatile Vec3<T> &operator=(volatile const Vec3 &v) volatile
    {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }
#endif

    FUNC const T *ptr() const { return &x; }

    FUNC T *ptr() { return &x; }

    FUNC const T &operator[](const size_t i) const
    {
        // assert(i < 3);
        return *(&x + i);
    }

    FUNC T &operator[](const size_t i)
    {
        // assert(i < 3);
        return *(&x + i);
    }

    FUNC Vec3 &operator+=(const Vec3 &other)
    {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    FUNC Vec3 &operator-=(const Vec3 &other)
    {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    FUNC Vec3 &operator*=(const T scalar)
    {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    FUNC Vec3 &operator/=(const T scalar)
    {
        // assert(scalar != 0);
        T inv = 1 / scalar;
        x *= inv;
        y *= inv;
        z *= inv;
        return *this;
    }

    FUNC bool operator==(Vec3 other) { return x == other.x && y == other.y && z == other.z; }
    FUNC bool operator!=(Vec3 other) { return !(*this == other); }

public:
    T x, y, z;
};

template <typename T> FUNC Vec3<T> operator+(const Vec3<T> &vec1, const Vec3<T> &vec2)
{
    return Vec3<T>(vec1.x + vec2.x, vec1.y + vec2.y, vec1.z + vec2.z);
}

template <typename T> FUNC Vec3<T> operator-(const Vec3<T> &vec1, const Vec3<T> &vec2)
{
    return Vec3<T>(vec1.x - vec2.x, vec1.y - vec2.y, vec1.z - vec2.z);
}

template <typename T> FUNC Vec3<T> operator-(const Vec3<T> &vec)
{
    return Vec3<T>(-vec.x, -vec.y, -vec.z);
}

template <typename T> FUNC Vec3<T> operator*(const Vec3<T> &vec, T scalar)
{
    return Vec3<T>(vec.x * scalar, vec.y * scalar, vec.z * scalar);
}

template <typename T> FUNC Vec3<T> operator*(T scalar, const Vec3<T> &vec)
{
    return Vec3<T>(vec.x * scalar, vec.y * scalar, vec.z * scalar);
}

template <typename T> FUNC Vec3<T> operator/(const Vec3<T> &vec, T scalar)
{
    // assert(scalar != 0);
    T inv = 1 / scalar;
    return Vec3<T>(vec.x * inv, vec.y * inv, vec.z * inv);
}

template <typename T> FUNC Vec3<T> mul(const Vec3<T> &vec1, const Vec3<T> &vec2)
{
    return Vec3<T>(vec1.x * vec2.x, vec1.y * vec2.y, vec1.z * vec2.z);
}

template <typename T> FUNC Vec3<T> div(const Vec3<T> &vec1, const Vec3<T> &vec2)
{
    return Vec3<T>(vec1.x / vec2.x, vec1.y / vec2.y, vec1.z / vec2.z);
}

template <typename T> FUNC Vec3<T> cross(const Vec3<T> &vec1, const Vec3<T> &vec2)
{
    return Vec3<T>(vec1.y * vec2.z - vec1.z * vec2.y, vec1.z * vec2.x - vec1.x * vec2.z,
                   vec1.x * vec2.y - vec1.y * vec2.x);
}

template <typename T> FUNC T dot(const Vec3<T> &vec1, const Vec3<T> &vec2)
{
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

template <typename T> FUNC Vec3<T> normalize(const Vec3<T> &vec)
{
    // assert(!(vec.x == 0 && vec.y == 0 && vec.z == 0));
    T inv = 1 / std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
    return Vec3<T>(vec.x * inv, vec.y * inv, vec.z * inv);
}

template <> FUNC Vec3<float> normalize(const Vec3<float> &vec)
{
// assert(!(vec.x == 0 && vec.y == 0 && vec.z == 0));
#ifdef __CUDACC__
    float inv = rsqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
#else
    auto inv = 1 / sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
#endif
    return Vec3<float>(vec.x * inv, vec.y * inv, vec.z * inv);
}

template <typename T> FUNC T lengthSquared(const Vec3<T> &v)
{
    return (v.x * v.x + v.y * v.y + v.z * v.z);
}

template <typename T> FUNC T length(const Vec3<T> &v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

template <typename T> FUNC const Vec3<T> abs(Vec3<T> v)
{
    return Vec3<T>(abs(v.x), abs(v.y), abs(v.z));
}

using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;
using Vec3u = Vec3<uint32_t>;
using Vec3i = Vec3<int>;

template <> FUNC const Vec3f min(const Vec3f a, const Vec3f b)
{
    return Vec3f(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

template <> FUNC const Vec3f max(const Vec3f a, const Vec3f b)
{
    return Vec3f(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

template <typename T> class Vec4
{
public:
    Vec4() = default;

    FUNC Vec4(T p1, T p2, T p3, T p4)
        : x(p1)
        , y(p2)
        , z(p3)
        , w(p4)
    {
    }

    FUNC Vec4(const Vec3<T> &v, T vW)
        : x(v.x)
        , y(v.y)
        , z(v.z)
        , w(vW)
    {
    }

    FUNC Vec4(const Vec3<T> &v)
        : x(v.x)
        , y(v.y)
        , z(v.z)
        , w(0)
    {
    }

    FUNC Vec4(const T *arr)
        : x(arr[0])
        , y(arr[1])
        , z(arr[2])
        , w(arr[3])
    {
    }

    FUNC explicit Vec4(T scalar)
        : x(scalar)
        , y(scalar)
        , z(scalar)
        , w(scalar)
    {
    }

#ifdef CUDA_SUPPORT
    FUNC explicit Vec4(float4 vec)
        : x(vec.x)
        , y(vec.y)
        , z(vec.z)
        , w(vec.w)
    {
    }
#endif

    FUNC const T *ptr() const { return &x; }

    FUNC T *ptr() { return &x; }

    FUNC const T &operator[](const size_t i) const
    {
        // assert(i < 4);
        return *(&x + i);
    }

    FUNC T &operator[](const size_t i)
    {
        // assert(i < 4);
        return *(&x + i);
    }

    FUNC Vec4 &operator+=(const Vec4 &other)
    {
        x += other.x;
        y += other.y;
        z += other.z;
        w += other.w;
        return *this;
    }

    FUNC Vec4 &operator-=(const Vec4 &other)
    {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        w -= other.w;
        return *this;
    }

    FUNC Vec4 &operator*=(const T scalar)
    {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        w *= scalar;
        return *this;
    }

    FUNC Vec4 &operator/=(const T scalar)
    {
        // assert(scalar != 0);
        T inv = 1 / scalar;
        x *= inv;
        y *= inv;
        z *= inv;
        w *= inv;
        return *this;
    }

public:
    T x, y, z, w;
};

template <typename T> FUNC const Vec4<T> abs(Vec4<T> v)
{
    return Vec4<T>(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

template <typename T> FUNC Vec4<T> operator+(const Vec4<T> &vec1, const Vec4<T> &vec2)
{
    return Vec4<T>(vec1.x + vec2.x, vec1.y + vec2.y, vec1.z + vec2.z, vec1.w + vec2.w);
}

template <typename T> FUNC Vec4<T> operator-(const Vec4<T> &vec1, const Vec4<T> &vec2)
{
    return Vec4<T>(vec1.x - vec2.x, vec1.y - vec2.y, vec1.z - vec2.z, vec1.w - vec2.w);
}

template <typename T> FUNC Vec4<T> operator-(const Vec4<T> &vec)
{
    return Vec4<T>(-vec.x, -vec.y, -vec.z, -vec.w);
}

template <typename T> FUNC Vec4<T> operator*(const Vec4<T> &vec, T scalar)
{
    return Vec4<T>(vec.x * scalar, vec.y * scalar, vec.z * scalar, vec.w * scalar);
}

template <typename T> FUNC Vec4<T> operator*(T scalar, const Vec4<T> &vec)
{
    return Vec4<T>(vec.x * scalar, vec.y * scalar, vec.z * scalar, vec.w * scalar);
}

template <typename T> FUNC Vec4<T> operator/(const Vec4<T> &vec, T scalar)
{
    // assert(scalar != 0);
    T inv = 1 / scalar;
    return Vec4<T>(vec.x * inv, vec.y * inv, vec.z * inv, vec.w * inv);
}

template <typename T> FUNC Vec4<T> mul(const Vec4<T> &vec1, const Vec4<T> &vec2)
{
    return Vec4<T>(vec1.x * vec2.x, vec1.y * vec2.y, vec1.z * vec2.z, vec1.w * vec2.w);
}

template <typename T> FUNC Vec4<T> div(const Vec4<T> &vec1, const Vec4<T> &vec2)
{
    return Vec4<T>(vec1.x / vec2.x, vec1.y / vec2.y, vec1.z / vec2.z, vec1.w / vec2.w);
}

template <typename T> FUNC T dot(const Vec4<T> &vec1, const Vec4<T> &vec2)
{
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z + vec1.w * vec2.w;
}

template <typename T> FUNC Vec4<T> normalize(const Vec4<T> &vec)
{
    // assert(!(vec.x == 0 && vec.y == 0 && vec.z == 0 && vec.w == 0));
    T inv = 1 / std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w);
    return Vec4<T>(vec.x * inv, vec.y * inv, vec.z * inv, vec.w * inv);
}

template <> FUNC Vec4<float> normalize(const Vec4<float> &vec)
{
#ifdef __CUDACC__
    float inv = rsqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w);
#else
    float inv = 1.0f / sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w);
#endif
    return Vec4<float>(vec.x * inv, vec.y * inv, vec.z * inv, vec.w * inv);
}

template <typename T> FUNC T lengthSquared(const Vec4<T> &v)
{
    return (v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
}

template <typename T> FUNC T length(const Vec4<T> &v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
}

using Vec4f = Vec4<float>;
using Vec4d = Vec4<double>;
using Vec4u = Vec4<uint32_t>;
using Vec4i = Vec4<int>;

template <> FUNC const Vec4f min(const Vec4f a, const Vec4f b)
{
    return Vec4f(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

template <> FUNC const Vec4f max(const Vec4f a, const Vec4f b)
{
    return Vec4f(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

// Explicit constructors

template <typename VEC> FUNC Vec2i make_vec2i(VEC v)
{
    return Vec2i(v.x, v.y);
}

template <typename VEC> FUNC Vec2f make_vec2f(VEC v)
{
    return Vec2f(v.x, v.y);
}

template <typename In, typename Out> FUNC Vec2<Out> cast(Vec2<In> v)
{
    return Vec2<Out>(v.x, v.y);
}

template <typename T> FUNC Vec2f make_vec2f(Vec2<T> v)
{
    return Vec2f(static_cast<float>(v.x), static_cast<float>(v.y));
}

template <typename T> FUNC Vec2f make_vec2f(const Vec3<T> &v)
{
    return Vec2f(v.x, v.y);
}

template <typename T> FUNC Vec2f make_vec2f(const Vec4<T> &v)
{
    return Vec2f(v.x, v.y);
}

#ifdef CUDA_SUPPORT
FUNC Vec2f make_vec2f(float4 v)
{
    return Vec2f(v.x, v.y);
}
#endif

template <typename In, typename Out> FUNC Vec3<Out> cast(Vec3<In> v)
{
    return Vec3<Out>(v.x, v.y, v.z);
}

template <typename VEC> FUNC Vec3f make_vec3f(const VEC v)
{
    return Vec3f(v.x, v.y, v.z);
}

template <typename VEC> FUNC Vec3d make_vec3d(const VEC v)
{
    return Vec3d(v.x, v.y, v.z);
}

template <typename In, typename Out> FUNC Vec4<Out> cast(Vec4<In> v)
{
    return Vec4<Out>(v.x, v.y, v.z, v.w);
}

template <typename VEC> FUNC Vec4f make_vec4f(const VEC v)
{
    return Vec4f(v.x, v.y, v.z, v.w);
}

template <typename T> FUNC T min3(T x, T y, T z)
{
    return min(min(x, y), z);
}

template <typename T> FUNC T max3(T x, T y, T z)
{
    return max(max(x, y), z);
}

/** Clamps the given value between min and max */
template <typename T> FUNC const T clamp(const T value, const T minValue, const T maxValue)
{
    return min(max(value, minValue), maxValue);
}

template <typename T> FUNC const Vec3<T> clamp(const Vec3<T> value, T minValue, T maxValue)
{
    auto x = min(max(value.x, minValue), maxValue);
    auto y = min(max(value.y, minValue), maxValue);
    auto z = min(max(value.z, minValue), maxValue);
    return Vec3<T>(x, y, z);
}

template <typename T> FUNC const Vec4<T> clamp(const Vec4<T> value, T minValue, T maxValue)
{
    auto x = min(max(value.x, minValue), maxValue);
    auto y = min(max(value.y, minValue), maxValue);
    auto z = min(max(value.z, minValue), maxValue);
    auto w = min(max(value.w, minValue), maxValue);
    return Vec4<T>(x, y, z, w);
}

// Computes a vector that is orthogonal to v.
template <typename T> FUNC Vec3<T> orthogonal(Vec3<T> v)
{
    if (length(v) == 0.0)
        return Vec3<T>(0, 0, 0);

    //    vec3 vR = normalize(v);
    //    float phi = atan(vR.y, vR.x);
    //    return -sin(phi) * vec3(1.0, 0.0, 0.0) + cos(phi) * vec3(0, 1.0, 0);

    if ((abs(v.y) >= 0.9f * abs(v.x)) && (abs(v.z) >= 0.9f * abs(v.x)))
        return Vec3<T>(0, -v.z, v.y);
    else if ((abs(v.x) >= 0.9f * abs(v.y)) && (abs(v.z) >= 0.9f * abs(v.y)))
        return Vec3<T>(-v.z, 0, v.x);
    else
        return Vec3<T>(-v.y, v.x, 0);
}

// Compute a righ-handed local frame of reference from vector w
template <typename T> FUNC void rhLocalFrame(Vec3<T> w, Vec3<T> &u, Vec3<T> &v)
{
    Vec3<T> outU = normalize(orthogonal(w));
    Vec3<T> outV = normalize(cross(outU, w));

    // Make sure that the system is right-handed
    float d = dot(cross(outU, outV), w);

    if (d < 0.0f)
    {
        outU.x = -outU.x;
        outU.y = -outU.y;
        outU.z = -outU.z;
    }

    u = outU;
    v = outV;
}

} // namespace cut

#endif
