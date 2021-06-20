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

#include "transfer_function.h"
#include <fstream>
#include <iostream>
#include "parameters.h"
#include "cut/gaussian.h"

inline Vec4f get(const array<Vec4f, TransferFunction::RESOLUTION> &tf1d, int i)
{
    i = cut::clamp(i, 0, TransferFunction::RESOLUTION - 1);
    return tf1d[i];
}

inline Vec4f sample1D(const array<Vec4f, TransferFunction::RESOLUTION> &tf1d, float x)
{
    float ip = x * static_cast<float>(TransferFunction::RESOLUTION);
    float a = ip - std::floor(ip);
    int xl = std::floor(ip);
    return (1.f - a) * get(tf1d, xl) + a * get(tf1d, xl + 1);
}

TransferFunction preintegrate(const array<Vec4f, TransferFunction::RESOLUTION> &tf1d, float step_size)
{
    TransferFunction tf;

#pragma omp parallel for
    for (int y = 0; y < TransferFunction::RESOLUTION; ++y)
    {
        for (int x = 0; x < TransferFunction::RESOLUTION; ++x)
        {
            float sf = x / float(TransferFunction::RESOLUTION);
            float sb = y / float(TransferFunction::RESOLUTION);

            // Number of steps to discretize the integral
            int num_steps = 8 + std::abs(y - x) / 2;

            float dw = 1.0f / float(num_steps);

            Vec4f rgba_sum(0, 0, 0, 0);
            for (int i = 0; i < num_steps; ++i)
            {
                float w = float(i) / float(num_steps - 1);

                // Linear interpolation of scalar value
                float s = w * sb + (1 - w) * sf;

                auto rgba = sample1D(tf1d, s);
                rgba.w *= step_size;

                rgba_sum.w += rgba.w * dw;

                float od = exp(-rgba_sum.w);
                rgba_sum.x += rgba.w * rgba.x * od * dw;
                rgba_sum.y += rgba.w * rgba.y * od * dw;
                rgba_sum.z += rgba.w * rgba.z * od * dw;
            }

            rgba_sum.w = 1.0f - std::exp(-rgba_sum.w);

            // tf.data[y * TransferFunction::RESOLUTION + x] = rgba_sum;

            Vec4f test = tf1d[x];
            test.w *= step_size;
            test.x = test.x * test.w;
            test.y = test.y * test.w;
            test.z = test.z * test.w;

            tf.data[y * TransferFunction::RESOLUTION + x] = test;
        }
    }

    return tf;
}

array<Vec4f, TransferFunction::RESOLUTION> read_tf(const char *filepath)
{
    std::ifstream file(filepath, std::ios::in | std::ios::binary);
    if (file.fail())
    {
        std::cerr << "Could not open file " << filepath << "!\n";
        throw invalid_args("Invalid transfer function file given!");
    }

    array<unsigned char, TransferFunction::RESOLUTION * 4> data;

    file.read(reinterpret_cast<char *>(&data[0]), TransferFunction::RESOLUTION * 4);
    file.close();

    array<Vec4f, TransferFunction::RESOLUTION> float_tf;
    for (int i = 0; i < TransferFunction::RESOLUTION; ++i)
        float_tf[i] = Vec4f(std::pow(data[i * 4] / 255.f, 2.2f), std::pow(data[i * 4 + 1] / 255.f, 2.2f),
                            std::pow(data[i * 4 + 2] / 255.f, 2.2f), data[i * 4 + 3] / 255.f);

    return float_tf;
}

TransferFunction read_from_disk_1D(const char *filepath)
{
    auto tf1D = read_tf(filepath);
    TransferFunction tf;
    for (int y = 0; y < TransferFunction::RESOLUTION; ++y)
    {
        for (int x = 0; x < TransferFunction::RESOLUTION; ++x)
        {
            tf.data[y * TransferFunction::RESOLUTION + x] = tf1D[x];
        }
    }
    return tf;
}

TransferFunction read_from_disk(const char *filepath, float step_size)
{
    return preintegrate(read_tf(filepath), step_size);
}

TransferFunction read_from_disk_interpolate(const char *filepath_1, const char *filepath_2, float t, float step_size)
{
    auto tf1 = read_tf(filepath_1);
    auto tf2 = read_tf(filepath_2);

    array<Vec4f, TransferFunction::RESOLUTION> tf;
    for (int i = 0; i < TransferFunction::RESOLUTION; ++i)
        tf[i] = cut::lerp(tf1[i], tf2[i], t);

    return preintegrate(tf, step_size);
}

TransferFunction box_filter(array<Vec4f, TransferFunction::RESOLUTION> tf1d, float step_size)
{
    TransferFunction tf;
#pragma omp parallel for default(none) shared(tf, step_size, tf1d)
    for (int y = 0; y < TransferFunction::RESOLUTION; ++y)
    {
        for (int x = 0; x < TransferFunction::RESOLUTION; ++x)
        {
            float sf = x / float(TransferFunction::RESOLUTION);
            float sb = y / float(TransferFunction::RESOLUTION);

            int num_steps = std::abs(y - x) + 2;
            float dw = 1.0f / float(num_steps);

            Vec4f rgba(0.0f);
            for (int i = 0; i < num_steps; ++i)
            {
                float w = float(i) / float(num_steps - 1);

                // Linear interpolation of scalar value
                float s = w * sb + (1 - w) * sf;

                auto tf = sample1D(tf1d, s);
                float a = dw * tf.w * step_size;
                rgba.x += tf.x * a;
                rgba.y += tf.y * a;
                rgba.z += tf.z * a;
                rgba.w += a;
            }

            tf.data[y * TransferFunction::RESOLUTION + x] = rgba;
        }
    }

    return tf;
}

TransferFunction gaussian_filter(array<Vec4f, TransferFunction::RESOLUTION> tf1d, float step_size)
{
    TransferFunction tf;
#pragma omp parallel for default(none) shared(tf, tf1d, step_size)
    for (int y = 0; y < TransferFunction::RESOLUTION; ++y)
    {
        for (int x = 0; x < TransferFunction::RESOLUTION; ++x)
        {
            float mean = x / float(TransferFunction::RESOLUTION);
            float std_dev = y / float(TransferFunction::RESOLUTION);

            Vec4f rgba(0.0f);
            if (std_dev != 0.0f)
            {
                float delta = 1.0f / static_cast<float>(TransferFunction::RESOLUTION);

                for (int i = 0; i < TransferFunction::RESOLUTION; ++i)
                {
                    float x = i * delta;
                    float weight = cut::normal_cdf(x + delta, mean, std_dev) - cut::normal_cdf(x, mean, std_dev);

                    float a = weight * tf1d[i].w * step_size;
                    rgba.x += tf1d[i].x * a;
                    rgba.y += tf1d[i].y * a;
                    rgba.z += tf1d[i].z * a;
                    rgba.w += a;
                }
            }
            else
            {
                rgba = tf1d[x];
                rgba.w *= step_size;
                rgba.x = rgba.x * rgba.w;
                rgba.y = rgba.y * rgba.w;
                rgba.z = rgba.z * rgba.w;
            }

            tf.data[y * TransferFunction::RESOLUTION + x] = rgba;
        }
    }

    return tf;
}

TransferFunction read_from_disk_uncertainty(const char *filepath, float step_size, int error_type)
{
    auto tf1d = read_tf(filepath);

    if (error_type == MAX_ERROR)
        return box_filter(tf1d, step_size);
    else if (error_type == RMSE)
        return gaussian_filter(tf1d, step_size);
    else if (error_type == ERROR_PERCENTILE)
        return box_filter(tf1d, step_size);
    else
    {
        assert(false);
        return TransferFunction();
    }
}
