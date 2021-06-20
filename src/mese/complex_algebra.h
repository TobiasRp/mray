// Copyright (c) 2019, Christoph Peters
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
#ifndef COMPLEX_ALGEBRA_H
#define COMPLEX_ALGEBRA_H

/*! \file
    This header defines utility functions to deal with complex numbers.*/

/*! A type for complex numbers that is not std::complex<float> (because for
    some reason, arithmetic operations on this type do not FUNC properly
    in some important implementations of the standard library) */
struct float_complex
{
    //! Real part and imaginary part
    float x, y;

    //! The default constructor leaves values uninitialized
    FUNC float_complex() {}
    FUNC float_complex(float real)
        : x(real)
        , y(0.0f)
    {
    }
    //! Initializes with the given real and imaginary part
    FUNC float_complex(float real, float imaginary)
        : x(real)
        , y(imaginary)
    {
    }
};

/*!	Returns the complex conjugate of the given complex number (i.e. it changes
    the sign of the y-component).*/
FUNC float_complex conjugate(float_complex z)
{
    return float_complex(z.x, -z.y);
}
/*!	Component-wise sign flip.*/
FUNC float_complex operator-(float_complex z)
{
    return float_complex(-z.x, -z.y);
}
/*!	This operator implements complex addition.*/
FUNC float_complex operator+(float_complex lhs, float_complex rhs)
{
    return float_complex(lhs.x + rhs.x, lhs.y + rhs.y);
}
/*!	This operator implements complex subtraction.*/
FUNC float_complex operator-(float_complex lhs, float_complex rhs)
{
    return float_complex(lhs.x - rhs.x, lhs.y - rhs.y);
}
/*!	This operator implements complex multiplication.*/
FUNC float_complex operator*(float_complex lhs, float_complex rhs)
{
    return float_complex(lhs.x * rhs.x - lhs.y * rhs.y, lhs.x * rhs.y + lhs.y * rhs.x);
}
/*!	This operator implements mixed real complex addition.*/
FUNC float_complex operator+(float lhs, float_complex rhs)
{
    return float_complex(lhs + rhs.x, rhs.y);
}
/*!	This operator implements  mixed real complex multiplication.*/
FUNC float_complex operator*(float lhs, float_complex rhs)
{
    return float_complex(lhs * rhs.x, lhs * rhs.y);
}
/*!	This function computes the squared magnitude of the given complex number.*/
FUNC float absSqr(float_complex z)
{
    return z.x * z.x + z.y * z.y;
}
/*!	This function computes the squared magnitude of the given complex number.*/
FUNC float sqr(float z)
{
    return z * z;
}
/*!	This operator computes the quotient of two complex numbers. The denominator
    must not be zero.*/
FUNC float_complex operator/(float_complex numerator, float_complex denominator)
{
    float factor = 1.0f / absSqr(denominator);
    return float_complex((numerator.x * denominator.x + numerator.y * denominator.y) * factor,
                         (-numerator.x * denominator.y + numerator.y * denominator.x) * factor);
}
/*!	This operator computes the quotient of a complex numerator and a real
    denominator.*/
FUNC float_complex operator/(float_complex numerator, float denominator)
{
    float factor = 1.0f / denominator;
    return float_complex(numerator.x * factor, numerator.y * factor);
}
FUNC bool operator==(float_complex lhs, float_complex rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y;
}
FUNC bool operator!=(float_complex lhs, float_complex rhs)
{
    return !(lhs == rhs);
}
/*!	This function implements computation of the reciprocal of the given
    non-zero complex number.*/
FUNC float_complex reciprocal(float_complex z)
{
    float factor = 1.0f / absSqr(z);
    return float_complex(z.x * factor, -z.y * factor);
}

/*! Implements an approximation to arctan with a maximal absolute error around
    1.1e-5f.*/
FUNC float fast_atan(float tan)
{
    int negative = (tan < 0.0f);
    float abs_tan = negative ? (-tan) : tan;
    int greater_one = (abs_tan > 1.0f);
    float x = greater_one ? abs_tan : 1.0f;
    x = 1.0f / x;
    float y = greater_one ? 1.0f : abs_tan;
    x = x * y;
    y = x * x;
    float z = y * 0.020835f - 0.085133f;
    z = y * z + 0.180141f;
    z = y * z - 0.330299f;
    y = y * z + 0.999866f;
    z = y * x;
    z = z * -2.0f + 1.570796f;
    z = greater_one ? z : 0.0f;
    x = x * y + z;
    return negative ? (-x) : x;
}

/*! Eulers formula for the complex exponential exp(i x) */
FUNC float_complex euler_exp(float x)
{
    return float_complex(cos(x), sin(x));
}

#endif
