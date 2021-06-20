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
#ifndef MRAY_ERROR_MEASURES_H
#define MRAY_ERROR_MEASURES_H

#include "common.h"

struct ErrorRMSE
{
    float squares = 0.0f;
    int count = 0;

    FUNC void update(float ref, float pred)
    {
        count += 1;
        auto diff = (pred - ref);
        squares += diff * diff;
    }
    FUNC float error() const { return sqrtf(squares / static_cast<float>(count)); }
};

struct ErrorMax
{
    float max_error = 0.0f;
    FUNC void update(float ref, float pred) { max_error = cut::max(max_error, cut::abs(pred - ref)); }
    FUNC float error() const { return max_error; }
};

template <typename T> FUNC void myswap(T &a, T &b)
{
    T temp;
    temp = a;
    a = b;
    b = temp;
}

template <typename T> FUNC void sort_five_numbers(T *v)
{
    if (v[1] < v[0])
        myswap(v[0], v[1]);
    if (v[2] < v[1])
    {
        myswap(v[2], v[1]);
        if (v[1] < v[0])
            myswap(v[1], v[0]);
    }
    if (v[3] < v[2])
    {
        myswap(v[3], v[2]);
        if (v[1] < v[0])
        {
            myswap(v[1], v[0]);
        }
        if (v[2] < v[1])
        {
            myswap(v[2], v[1]);
            if (v[1] < v[0])
                myswap(v[1], v[0]);
        }
    }
    if (v[4] < v[3])
    {
        myswap(v[4], v[3]);
        if (v[3] < v[2])
        {
            myswap(v[3], v[2]);
            if (v[1] < v[0])
            {
                myswap(v[1], v[0]);
            }
            if (v[2] < v[1])
            {
                myswap(v[2], v[1]);
                if (v[1] < v[0])
                    myswap(v[1], v[0]);
            }
        }
    }
}

/**
 * The P-square algorithm to estimate a percentile.
 * Cf. https://www.cse.wustl.edu/%7Ejain/papers/ftp/psqr.pdf
 */
struct ErrorPercentile
{
    float q;
    int init_num;

    float marker_heights[5];
    int marker_positions[5];
    float desired_positions[5];

    FUNC ErrorPercentile(float p)
        : q(p / 100.f)
        , init_num(0)
    {
    }

    FUNC float parabolic(int i, int d)
    {
        auto t1 = d / static_cast<float>(marker_positions[i + 1] - marker_positions[i - 1]);

        auto t2 = (marker_positions[i] - marker_positions[i - 1] + d) * (marker_heights[i + 1] - marker_heights[i]) /
                  static_cast<float>(marker_positions[i + 1] - marker_positions[i]);

        auto t3 = (marker_positions[i + 1] - marker_positions[i] - d) * (marker_heights[i] - marker_heights[i - 1]) /
                  static_cast<float>(marker_positions[i] - marker_positions[i - 1]);

        return marker_heights[i] + t1 * (t2 + t3);
    }

    FUNC float linear(int i, int d)
    {
        return marker_heights[i] + d * (marker_heights[i + d] - marker_heights[i]) /
                                       static_cast<float>(marker_positions[i + d] - marker_positions[i]);
    }

    FUNC void adjust_heights()
    {
        for (int i = 1; i < 5; ++i)
        {
            float di = desired_positions[i] - marker_positions[i];
            if ((di >= 1 && (marker_positions[i + 1] - marker_positions[i] > 1)) ||
                (di <= -1 && (marker_positions[i - 1] - marker_positions[i] < -1)))
            {
                int sdi = (di > 0) ? 1 : -1;

                auto qprime = parabolic(i, sdi);
                if ((marker_heights[i - 1] < qprime) && (qprime < marker_heights[i + 1]))
                    marker_heights[i] = qprime;
                else
                {
                    qprime = linear(i, sdi);
                    marker_heights[i] = qprime;
                }

                marker_positions[i] += sdi;
            }
        }
    }

    FUNC void update(float ref, float pred)
    {
        float value = cut::abs(pred - ref);
        if (init_num < 5)
        {
            marker_heights[init_num] = value;
            ++init_num;

            if (init_num == 5)
            {
                sort_five_numbers(marker_heights);

                for (int i = 0; i < 5; ++i)
                    marker_positions[i] = 1 + i;

                desired_positions[0] = 1.0f;
                desired_positions[1] = 1.0f + 2.0f * q;
                desired_positions[2] = 1.0f + 4.0f * q;
                desired_positions[3] = 3.0f + 2.0f * q;
                desired_positions[4] = 5.0f;
            }
            return;
        }

        int move_idx = 0;
        if (value < marker_heights[0])
        {
            marker_heights[0] = value;
            move_idx = 1;
        }
        else if (value < marker_heights[1])
        {
            move_idx = 1;
        }
        else if (value < marker_heights[2])
        {
            move_idx = 2;
        }
        else if (value < marker_heights[3])
        {
            move_idx = 3;
        }
        else if (value < marker_heights[4])
        {
            move_idx = 4;
        }
        else if (value > marker_heights[4])
        {
            marker_heights[4] = value;
            move_idx = 4;
        }

        for (int k = move_idx; k < 5; ++k)
            marker_positions[k] += 1;

        desired_positions[1] += q / 2.0f;
        desired_positions[2] += q;
        desired_positions[3] += (1.0f + q) / 2.f;
        desired_positions[4] += 1.0f;

        adjust_heights();
    }

    FUNC float error() const { return marker_heights[2]; }
};

#endif // MRAY_ERROR_MEASURES_H
