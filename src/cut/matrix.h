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
#ifndef MATRIX_H
#define MATRIX_H

#include "cuda_math.h"

namespace cut
{

/**
 * 2x2 matrix
 */
template <typename T> class Mat2
{
public:
    FUNC Mat2() {}

    /* Creates a matrix with s on the diagonal */
    FUNC Mat2(T s)
    {
        m[0][0] = s;
        m[0][1] = 0;
        m[1][1] = s;
        m[1][0] = 0;
    }

    FUNC Mat2(T m00, T m01, T m10, T m11)
    {
        m[0][0] = m00;
        m[0][1] = m01;
        m[1][0] = m10;
        m[1][1] = m11;
    }

    /** Creates a 2x2 matrix from an array.
     * \note Make sure that the array uses the same memory layout as this matrix.
     */
    FUNC Mat2(const T *arr) { memcpy(m, arr, 4 * sizeof(T)); }

    FUNC const T *operator[](uint32_t row) const { return m[row]; }

    FUNC T *operator[](uint32_t row) { return m[row]; }

    FUNC Mat2<T> &operator=(const Mat2 &mat)
    {
        memcpy(m, mat.m, sizeof(T) * 4);
        return *this;
    }

    FUNC T *ptr() { return reinterpret_cast<T *>(&m); }

    FUNC T determinant() const { return m[0][0] * m[1][1] - m[0][1] * m[1][0]; }

    FUNC Mat2 operator*(T s) { return Mat2<T>(m[0][0] * s, m[0][1] * s, m[1][0] * s, m[1][1] * s); }

    FUNC Mat2 &operator*=(T s)
    {
        m[0][0] *= s;
        m[0][1] *= s;
        m[1][0] *= s;
        m[1][1] *= s;
        return *this;
    }

    FUNC Mat2 &operator+=(Mat2<T> A)
    {
        m[0][0] += A.m[0][0];
        m[0][1] += A.m[0][1];

        m[1][0] += A.m[1][0];
        m[1][1] += A.m[1][1];

        return *this;
    }

    FUNC bool operator==(Mat2<T> o)
    {
        return m[0][0] == o.m[0][0] && m[0][1] == o.m[0][1] && m[1][0] == o.m[1][0] && m[1][1] == o.m[1][1];
    }

    FUNC bool operator!=(Mat2<T> o) { return !operator==(o); }

private:
    T m[2][2];
};

using Mat2f = Mat2<float>;
using Mat2d = Mat2<double>;
using Mat2u = Mat2<uint32_t>;
using Mat2i = Mat2<int>;

template <typename In, typename Out> FUNC Mat2<Out> cast(Mat2<In> m)
{
    return Mat2<Out>(m[0][0], m[0][1], m[1][0], m[1][1]);
}

template <typename T> FUNC Mat2<T> transpose(Mat2<T> m)
{
    return Mat2<T>{m[0][0], m[1][0], m[0][1], m[1][1]};
}

template <typename T> FUNC Mat2<T> operator*(Mat2<T> mat1, Mat2<T> mat2)
{
    T m00 = mat1[0][0] * mat2[0][0] + mat1[0][1] * mat2[1][0];
    T m01 = mat1[0][0] * mat2[0][1] + mat1[0][1] * mat2[1][1];

    T m10 = mat1[1][0] * mat2[0][0] + mat1[1][1] * mat2[1][0];
    T m11 = mat1[1][0] * mat2[0][1] + mat1[1][1] * mat2[1][1];

    return Mat2<T>(m00, m01, m10, m11);
}

template <typename T> FUNC Mat2<T> operator*(const Mat2<T> &mat, const T s)
{
    return Mat2<T>(mat[0][0] * s, mat[0][1] * s, mat[1][0] * s, mat[1][1] * s);
}
template <typename T> FUNC Mat2<T> operator*(const T s, const Mat2<T> &mat)
{
    return Mat2<T>(mat[0][0] * s, mat[0][1] * s, mat[1][0] * s, mat[1][1] * s);
}

template <typename T> FUNC Mat2<T> operator+(const Mat2<T> &A, const Mat2<T> &B)
{
    T m00 = B[0][0] + A[0][0];
    T m01 = B[0][1] + A[0][1];
    T m10 = B[1][0] + A[1][0];
    T m11 = B[1][1] + A[1][1];
    return Mat2<T>(m00, m01, m10, m11);
}

template <typename T> FUNC Mat2<T> operator-(const Mat2<T> &A, const Mat2<T> &B)
{
    T m00 = A[0][0] - B[0][0];
    T m01 = A[0][1] - B[0][1];
    T m10 = A[1][0] - B[1][0];
    T m11 = A[1][1] - B[1][1];
    return Mat2<T>(m00, m01, m10, m11);
}

/** Multiplies the floating point vector from right to left, i.e. mat * vec
 */
template <typename T> FUNC Vec2<T> operator*(Mat2<T> mat, Vec2<T> vec)
{
    return Vec2<T>(mat[0][0] * vec.x + mat[0][1] * vec.y, mat[1][0] * vec.x + mat[1][1] * vec.y);
}

template <typename T> FUNC Mat2<T> inverse(Mat2<T> A)
{
    auto invDetA = 1.0 / A.determinant();

    return Mat2<T>(A[1][1] * invDetA, -A[0][1] * invDetA, -A[1][0] * invDetA, A[0][0] * invDetA);
}

/** Computes the inverse of a symmetric 2x2 matrix using the numerically
 * stable Cholesky decomposition */
template <typename T> FUNC Mat2<T> symmetric_inverse(Mat2<T> A)
{
    // Cholesky decomposition
    T a = std::sqrt(A[0][0]);
    T b = A[1][0] / a;
    T c = std::sqrt(A[1][1] - b * b);

    // Invert
    T l1 = (1.0 / a); //, - b / (a * c), (1.0 / c)
    T l2 = -b / (a * c);
    T l3 = (1.0 / c);

    // Multiply
    T x1 = l1 * l1 + l2 * l2;
    T x2 = l2 * l3;
    T x3 = l3 * l3;
    return cut::Mat2<T>(x1, x2, x2, x3);
}

template <typename T> FUNC Mat2<T> outerProduct(Vec2<T> a, Vec2<T> b)
{
    return Mat2<T>(a.x * b.x, a.x * b.y, a.y * b.x, a.y * b.y);
}

template <typename T> FUNC T trace(Mat2<T> M)
{
    return M[0][0] + M[1][1];
}

/**
 * 3x3 matrix
 */
template <typename T> class Mat3
{
public:
    FUNC Mat3() {}

    /* Creates a matrix with s on the diagonal */
    FUNC Mat3(T s)
    {
        memset(&m, 0, 9 * sizeof(T));
        m[0][0] = s;
        m[1][1] = s;
        m[2][2] = s;
    }

    FUNC Mat3(T m00, T m01, T m02, T m10, T m11, T m12, T m20, T m21, T m22)
    {
        m[0][0] = m00;
        m[0][1] = m01;
        m[0][2] = m02;
        m[1][0] = m10;
        m[1][1] = m11;
        m[1][2] = m12;
        m[2][0] = m20;
        m[2][1] = m21;
        m[2][2] = m22;
    }

    /** Creates a 3x3 matrix from an array.
     * \note Make sure that the array uses the same memory layout as this matrix.
     */
    FUNC Mat3(const T *arr) { memcpy(m, arr, 9 * sizeof(T)); }

    FUNC Mat3(const Mat3 &mat) { memcpy(m, mat.m, sizeof(T) * 9); }

    FUNC const T *operator[](uint32_t row) const { return m[row]; }

    FUNC T *operator[](uint32_t row) { return m[row]; }

    FUNC Mat3<T> &operator=(const Mat3 &mat)
    {
        memcpy(m, mat.m, sizeof(T) * 9);
        return *this;
    }

    FUNC T *ptr() { return reinterpret_cast<T *>(&m); }

    FUNC T determinant() const
    {
        return m[0][0] * m[1][1] * m[2][2] + m[0][1] * m[1][2] * m[2][0] + m[0][2] * m[1][0] * m[2][1] -
               m[0][2] * m[1][1] * m[2][0] - m[1][0] * m[0][1] * m[2][2] - m[0][0] * m[1][2] * m[2][1];
    }

    FUNC Mat3 operator*(T s)
    {
        return Mat3<T>(m[0][0] * s, m[0][1] * s, m[0][2] * s, m[1][0] * s, m[1][1] * s, m[1][2] * s, m[2][0] * s,
                       m[2][1] * s, m[2][2] * s);
    }

    FUNC Mat3 &operator*=(T s)
    {
        m[0][0] *= s;
        m[0][1] *= s;
        m[0][2] *= s;
        m[1][0] *= s;
        m[1][1] *= s;
        m[1][2] *= s;
        m[2][0] *= s;
        m[2][1] *= s;
        m[2][2] *= s;
        return *this;
    }

    FUNC Mat3 &operator+=(Mat3<T> A)
    {
        m[0][0] += A.m[0][0];
        m[0][1] += A.m[0][1];
        m[0][2] += A.m[0][2];

        m[1][0] += A.m[1][0];
        m[1][1] += A.m[1][1];
        m[1][2] += A.m[1][2];

        m[2][0] += A.m[2][0];
        m[2][1] += A.m[2][1];
        m[2][2] += A.m[2][2];
        return *this;
    }

    FUNC bool operator==(Mat3<T> o)
    {
        return m[0][0] == o.m[0][0] && m[0][1] == o.m[0][1] && m[1][0] == o.m[1][0] && m[1][1] == o.m[1][1] &&
               m[0][2] == o.m[0][2] && m[2][0] == o.m[2][0] && m[1][2] == o.m[1][2] && m[2][1] == o.m[2][1] &&
               m[2][2] == o.m[2][2];
    }

    FUNC bool operator!=(Mat3<T> o) { return !operator==(o); }

private:
    T m[3][3];
};

using Mat3f = Mat3<float>;
using Mat3d = Mat3<double>;
using Mat3u = Mat3<uint32_t>;
using Mat3i = Mat3<int>;

template <typename In, typename Out> FUNC Mat3<Out> cast(Mat3<In> m)
{
    return Mat3<Out>(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]);
}

template <typename T> FUNC Mat3<T> transpose(const Mat3<T> m)
{
    return Mat3<T>{m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1], m[0][2], m[1][2], m[2][2]};
}

template <typename T> FUNC Mat3<T> operator*(const Mat3<T> mat1, const Mat3<T> &mat2)
{
    T m00 = mat1[0][0] * mat2[0][0] + mat1[0][1] * mat2[1][0] + mat1[0][2] * mat2[2][0];
    T m01 = mat1[0][0] * mat2[0][1] + mat1[0][1] * mat2[1][1] + mat1[0][2] * mat2[2][1];
    T m02 = mat1[0][0] * mat2[0][2] + mat1[0][1] * mat2[1][2] + mat1[0][2] * mat2[2][2];

    T m10 = mat1[1][0] * mat2[0][0] + mat1[1][1] * mat2[1][0] + mat1[1][2] * mat2[2][0];
    T m11 = mat1[1][0] * mat2[0][1] + mat1[1][1] * mat2[1][1] + mat1[1][2] * mat2[2][1];
    T m12 = mat1[1][0] * mat2[0][2] + mat1[1][1] * mat2[1][2] + mat1[1][2] * mat2[2][2];

    T m20 = mat1[2][0] * mat2[0][0] + mat1[2][1] * mat2[1][0] + mat1[2][2] * mat2[2][0];
    T m21 = mat1[2][0] * mat2[0][1] + mat1[2][1] * mat2[1][1] + mat1[2][2] * mat2[2][1];
    T m22 = mat1[2][0] * mat2[0][2] + mat1[2][1] * mat2[1][2] + mat1[2][2] * mat2[2][2];

    return Mat3<T>(m00, m01, m02, m10, m11, m12, m20, m21, m22);
}

template <typename T> FUNC Mat3<T> operator+(const Mat3<T> &A, const Mat3<T> &B)
{
    T m00 = B[0][0] + A[0][0];
    T m01 = B[0][1] + A[0][1];
    T m02 = B[0][2] + A[0][2];

    T m10 = B[1][0] + A[1][0];
    T m11 = B[1][1] + A[1][1];
    T m12 = B[1][2] + A[1][2];

    T m20 = B[2][0] + A[2][0];
    T m21 = B[2][1] + A[2][1];
    T m22 = B[2][2] + A[2][2];

    return Mat3<T>(m00, m01, m02, m10, m11, m12, m20, m21, m22);
}

/** Multiplies the floating point vector from right to left, i.e. mat * vec
 */
template <typename T> FUNC Vec3<T> operator*(const Mat3<T> &mat, const Vec3<T> &vec)
{
    return Vec3<T>(mat[0][0] * vec.x + mat[0][1] * vec.y + mat[0][2] * vec.z,
                   mat[1][0] * vec.x + mat[1][1] * vec.y + mat[1][2] * vec.z,
                   mat[2][0] * vec.x + mat[2][1] * vec.y + mat[2][2] * vec.z);
}

template <typename T> FUNC Mat3<T> operator*(const Mat3<T> &mat, const T s)
{
    return Mat3<T>(mat[0][0] * s, mat[0][1] * s, mat[0][2] * s, mat[1][0] * s, mat[1][1] * s, mat[1][2] * s,
                   mat[2][0] * s, mat[2][1] * s, mat[2][2] * s);
}

template <typename T> FUNC Mat3<T> operator*(const T s, const Mat3<T> &mat)
{
    return Mat3<T>(mat[0][0] * s, mat[0][1] * s, mat[0][2] * s, mat[1][0] * s, mat[1][1] * s, mat[1][2] * s,
                   mat[2][0] * s, mat[2][1] * s, mat[2][2] * s);
}

template <typename T> FUNC Mat3<T> operator-(const Mat3<T> &A, const Mat3<T> &B)
{
    return Mat3<T>(A[0][0] - B[0][0], A[0][1] - B[0][1], A[0][2] - B[0][2], A[1][0] - B[1][0], A[1][1] - B[1][1],
                   A[1][2] - B[1][2], A[2][0] - B[2][0], A[2][1] - B[2][1], A[2][2] - B[2][2]);
}

template <typename T> FUNC Mat3<T> outerProduct(Vec3<T> a, Vec3<T> b)
{
    return Mat3<T>(a.x * b.x, a.x * b.y, a.x * b.z, a.y * b.x, a.y * b.y, a.y * b.z, a.z * b.x, a.z * b.y, a.z * b.z);
}

template <typename T> FUNC Mat3<T> inverse(const Mat3<T> &A)
{
    auto invDetA = 1.0 / A.determinant();

    Mat3<T> R;
    R[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * invDetA;
    R[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * invDetA;
    R[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * invDetA;

    R[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * invDetA;
    R[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * invDetA;
    R[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * invDetA;

    R[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * invDetA;
    R[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * invDetA;
    R[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * invDetA;

    return R;
}

template <typename T> FUNC T trace(Mat3<T> M)
{
    return M[0][0] + M[1][1] + M[2][2];
}

template <typename T> FUNC void eigenvaluesOfMat(Mat2<T> A, T &e1, T &e2)
{
    double trace = static_cast<double>(A[0][0]) + static_cast<double>(A[1][1]);

    double det = static_cast<double>(A[0][0]) * static_cast<double>(A[1][1]) -
                 static_cast<double>(A[1][0]) * static_cast<double>(A[1][0]);

    double t = ((trace * trace) / 4.0);
    double disc = t - det;

    if (!(disc >= 0.0))
    {
        e1 = 0.0;
        e2 = 0.0;
    }
    else
    {
        assert(disc >= 0.0);

        double root = sqrt(disc);
        e1 = 0.5 * trace + root;
        e2 = 0.5 * trace - root;
    }
}

template <typename T> FUNC void eigenvectorsOfMat(Mat2<T> A, T e1, T e2, Vec2<T> &v1, Vec2<T> &v2)
{
    if (abs(A[1][0]) > 1e-6)
    {
        v1 = Vec2<T>(e1 - A[1][1], A[1][0]);
        v2 = Vec2<T>(e2 - A[1][1], A[1][0]);
    }
    else if (abs(A[0][1]) > 1e-6)
    {
        v1 = Vec2<T>(A[0][1], e1 - A[0][0]);
        v2 = Vec2<T>(A[0][1], e2 - A[0][0]);
    }
    else
    {
        v1 = Vec2<T>(1, 0);
        v2 = Vec2<T>(0, 1);
    }
}

template <typename T> FUNC void eigenvaluesOfSymmetricMat(Mat3<T> A, T &e1, T &e2, T &e3)
{

    // http://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
    T p1 = A[0][1] * A[0][1] + A[0][2] * A[0][2] + A[1][2] * A[1][2];
    if (p1 == 0.0)
    {
        e1 = A[0][0];
        e2 = A[1][1];
        e3 = A[2][2];
    }
    else
    {
        T traceA = A[0][0] + A[1][1] + A[2][2];
        T q = traceA / 3.0;

        T p2 = (A[0][0] - q) * (A[0][0] - q) + (A[1][1] - q) * (A[1][1] - q) + (A[2][2] - q) * (A[2][2] - q) + 2.0 * p1;
        T p = sqrt(p2 / 6.0);

        Mat3<T> I(1.0);
        Mat3<T> B = (A - (I * q)) * (1.0 / p);

        T r = B.determinant() / 2.0;

        T phi = 0.0;
        if (r <= -1)
            phi = M_PI / 3.0;
        else if (r >= 1)
            phi = 0.0;
        else
            phi = acos(r) / 3.0;

        e1 = q + 2.0 * p * cos(phi);
        e3 = q + 2.0 * p * cos(phi + (2.0 * M_PI / 3.0f));
        e2 = 3.0 * q - e1 - e3;
    }
}

template <typename T>
FUNC void eigenvectorsOfSymmetricMat(Mat3<T> A, T e1, T e2, T e3, Vec3<T> &v1, Vec3<T> &v2, Vec3<T> &v3)
{
    // http://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
    Mat3<T> I{1.0};

    auto M1 = A - (I * e1); // matMatDifference(A, matScalarProduct(I, e1));
    auto M2 = A - (I * e2); // matMatDifference(A, matScalarProduct(I, e2));
    auto M3 = A - (I * e3); // matMatDifference(A, matScalarProduct(I, e3));

    auto M12 = M1 * M2; // matMatProduct(M1, M2);
    auto M13 = M1 * M3; // matMatProduct(M1, M3);
    auto M23 = M2 * M2; // matMatProduct(M2, M2);

    // first eigenvector
    Vec3<T> M23_1(M23[0][0], M23[1][0], M23[2][0]);
    Vec3<T> M23_2(M23[0][1], M23[1][1], M23[2][1]);
    Vec3<T> M23_3(M23[0][2], M23[1][2], M23[2][2]);

    T M23_1_len = length(M23_1);
    T M23_2_len = length(M23_2);
    T M23_3_len = length(M23_3);
    T max_len = max(max(M23_1_len, M23_2_len), M23_3_len);

    if (max_len == M23_1_len && M23_1_len > 0.0f)
        v1 = normalize(M23_1);
    else if (max_len == M23_2_len && M23_2_len > 0.0f)
        v1 = normalize(M23_2);
    else if (max_len == M23_3_len && M23_3_len > 0.0f)
        v1 = normalize(M23_3);

    // second eigenvector
    Vec3<T> M13_1(M13[0][0], M13[1][0], M13[2][0]);
    Vec3<T> M13_2(M13[0][1], M13[1][1], M13[2][1]);
    Vec3<T> M13_3(M13[0][2], M13[1][2], M13[2][2]);

    T M13_1_len = length(M13_1);
    T M13_2_len = length(M13_2);
    T M13_3_len = length(M13_3);
    max_len = max(max(M13_1_len, M13_2_len), M13_3_len);

    if (max_len == M13_1_len && M13_1_len > 0.0f)
        v2 = normalize(M13_1);
    else if (max_len == M13_2_len && M13_2_len > 0.0f)
        v2 = normalize(M13_2);
    else if (max_len == M13_3_len && M13_3_len > 0.0f)
        v2 = normalize(M13_3);

    // third eigenvector
    Vec3<T> M12_1(M12[0][0], M12[1][0], M12[2][0]);
    Vec3<T> M12_2(M12[0][1], M12[1][1], M12[2][1]);
    Vec3<T> M12_3(M12[0][2], M12[1][2], M12[2][2]);

    T M12_1_len = length(M12_1);
    T M12_2_len = length(M12_2);
    T M12_3_len = length(M12_3);
    max_len = max(max(M12_1_len, M12_2_len), M12_3_len);

    if (max_len == M12_1_len && M12_1_len > 0.0f)
        v3 = normalize(M12_1);
    else if (max_len == M12_2_len && M12_2_len > 0.0f)
        v3 = normalize(M12_2);
    else if (max_len == M12_3_len && M12_3_len > 0.0f)
        v3 = normalize(M12_3);

    return;
}

/** 4x4 matrix */
template <typename T> class Mat4
{
public:
    FUNC Mat4() {}

    /* Creates a matrix with s on the diagonal */
    FUNC Mat4(T s)
    {
        memset(&m, 0, 16 * sizeof(T));
        m[0][0] = s;
        m[1][1] = s;
        m[2][2] = s;
        m[3][3] = s;
    }

    FUNC Mat4(T m00, T m01, T m02, T m03, T m10, T m11, T m12, T m13, T m20, T m21, T m22, T m23, T m30, T m31, T m32,
              T m33)
    {
        m[0][0] = m00;
        m[0][1] = m01;
        m[0][2] = m02;
        m[0][3] = m03;
        m[1][0] = m10;
        m[1][1] = m11;
        m[1][2] = m12;
        m[1][3] = m13;
        m[2][0] = m20;
        m[2][1] = m21;
        m[2][2] = m22;
        m[2][3] = m23;
        m[3][0] = m30;
        m[3][1] = m31;
        m[3][2] = m32;
        m[3][3] = m33;
    }

    /** Creates a 4x4 matrix from an array.
     * \note Make sure that the array uses the same memory layout as this matrix.
     * GLM's mat4 is transposed in regard to this class!!!
     */
    FUNC Mat4(const T *arr) { memcpy(m, arr, 16 * sizeof(T)); }

    FUNC const T *operator[](uint32_t row) const { return m[row]; }

    FUNC T *operator[](uint32_t row) { return m[row]; }

    /** Assigns the 4x4 matrix. */
    FUNC Mat4<T> &operator=(const Mat4 &mat)
    {
        memcpy(m, mat.m, sizeof(T) * 16);
        return *this;
    }

    FUNC T *ptr() { return reinterpret_cast<T *>(&m); }

private:
    T m[4][4];
};

/** Concatenates two matrices, i.e. mat1 * mat2.
 * \remarks Matrix multiplication is not commutative, so the parameter order matters!
 */
template <typename T> FUNC Mat4<T> operator*(const Mat4<T> &mat1, const Mat4<T> &mat2)
{
    T m00 = mat1[0][0] * mat2[0][0] + mat1[0][1] * mat2[1][0] + mat1[0][2] * mat2[2][0] + mat1[0][3] * mat2[3][0];
    T m01 = mat1[0][0] * mat2[0][1] + mat1[0][1] * mat2[1][1] + mat1[0][2] * mat2[2][1] + mat1[0][3] * mat2[3][1];
    T m02 = mat1[0][0] * mat2[0][2] + mat1[0][1] * mat2[1][2] + mat1[0][2] * mat2[2][2] + mat1[0][3] * mat2[3][2];
    T m03 = mat1[0][0] * mat2[0][3] + mat1[0][1] * mat2[1][3] + mat1[0][2] * mat2[2][3] + mat1[0][3] * mat2[3][3];

    T m10 = mat1[1][0] * mat2[0][0] + mat1[1][1] * mat2[1][0] + mat1[1][2] * mat2[2][0] + mat1[1][3] * mat2[3][0];
    T m11 = mat1[1][0] * mat2[0][1] + mat1[1][1] * mat2[1][1] + mat1[1][2] * mat2[2][1] + mat1[1][3] * mat2[3][1];
    T m12 = mat1[1][0] * mat2[0][2] + mat1[1][1] * mat2[1][2] + mat1[1][2] * mat2[2][2] + mat1[1][3] * mat2[3][2];
    T m13 = mat1[1][0] * mat2[0][3] + mat1[1][1] * mat2[1][3] + mat1[1][2] * mat2[2][3] + mat1[1][3] * mat2[3][3];

    T m20 = mat1[2][0] * mat2[0][0] + mat1[2][1] * mat2[1][0] + mat1[2][2] * mat2[2][0] + mat1[2][3] * mat2[3][0];
    T m21 = mat1[2][0] * mat2[0][1] + mat1[2][1] * mat2[1][1] + mat1[2][2] * mat2[2][1] + mat1[2][3] * mat2[3][1];
    T m22 = mat1[2][0] * mat2[0][2] + mat1[2][1] * mat2[1][2] + mat1[2][2] * mat2[2][2] + mat1[2][3] * mat2[3][2];
    T m23 = mat1[2][0] * mat2[0][3] + mat1[2][1] * mat2[1][3] + mat1[2][2] * mat2[2][3] + mat1[2][3] * mat2[3][3];

    T m30 = mat1[3][0] * mat2[0][0] + mat1[3][1] * mat2[1][0] + mat1[3][2] * mat2[2][0] + mat1[3][3] * mat2[3][0];
    T m31 = mat1[3][0] * mat2[0][1] + mat1[3][1] * mat2[1][1] + mat1[3][2] * mat2[2][1] + mat1[3][3] * mat2[3][1];
    T m32 = mat1[3][0] * mat2[0][2] + mat1[3][1] * mat2[1][2] + mat1[3][2] * mat2[2][2] + mat1[3][3] * mat2[3][2];
    T m33 = mat1[3][0] * mat2[0][3] + mat1[3][1] * mat2[1][3] + mat1[3][2] * mat2[2][3] + mat1[3][3] * mat2[3][3];

    return Mat4<T>(m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33);
}

/** Multiplies the floating point vector from right to left, i.e. mat * vec
 */
FUNC Vec4f operator*(const Mat4<float> &mat, const Vec4f &vec)
{
    return Vec4f(mat[0][0] * vec.x + mat[0][1] * vec.y + mat[0][2] * vec.z + mat[0][3] * vec.w,
                 mat[1][0] * vec.x + mat[1][1] * vec.y + mat[1][2] * vec.z + mat[1][3] * vec.w,
                 mat[2][0] * vec.x + mat[2][1] * vec.y + mat[2][2] * vec.z + mat[2][3] * vec.w,
                 mat[3][0] * vec.x + mat[3][1] * vec.y + mat[3][2] * vec.z + mat[3][3] * vec.w);
}

template <typename T> FUNC T trace(Mat4<T> M)
{
    return M[0][0] + M[1][1] + M[2][2] + M[3][3];
}

using Mat4f = Mat4<float>;
using Mat4u = Mat4<uint32_t>;
using Mat4i = Mat4<int>;
} // namespace cut

#endif // MATRIX_H
