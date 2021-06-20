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
#include "gtest/gtest.h"
#include "ray_histogram_image.h"
#include "raymarch_rayhistogram.h"
#include "ray_histogram_sampler.h"

#include <random>

TEST(test_rayhistogram, test_log_table)
{
    for (int i = 1; i < 256; ++i)
        ASSERT_NEAR(std::log2(i / 256.f), e_log2(i / 256.f), 5e-3);
}

TEST(test_rayhistogram, test_entropy_update)
{
    array<uint16_t, RayHistogramGenerationParameters::NUM_BINS> hist;
    std::fill(hist.begin(), hist.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, hist.size() - 1);

    float e = 0.0f;
    uint32_t sum = 0;
    for (int i = 0; i < 128; ++i)
    {
        auto b = dist(gen);
        e = update(e, hist.data(), sum, b);

        auto expected_e = entropy(hist.data(), hist.size());
        ASSERT_NEAR(e, expected_e, 1e-6f);
    }
}

TEST(test_rayhistogram, test_rayhistogram_image)
{
    int w = 2;
    int h = 2;
    RayHistogramImageHost img(w, h);

    vector<PixelRayHistogram> rayhists(w*h);

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            PixelRayHistogram rh;

            for (int i = 0; i < x; ++i)
            {
                rh.add_frustum(i);

                for (int b = 0; b < i; ++b)
                    rh.add_bin((i + b) % 128, (i + b));
            }

            rayhists[y * w + x] = rh;
        }
    }

    img.create_from(rayhists);

    write_rayhistogram_image("test_rayhist.bin", img, false);

    auto loaded_img = read_rayhistogram_image("test_rayhist.bin");

    ASSERT_EQ(img.width, loaded_img.width);
    ASSERT_EQ(img.height, loaded_img.height);

    for (size_t i = 0; i < img.indices.size(); ++i)
        ASSERT_EQ(img.indices[i], loaded_img.indices[i]);

    for (size_t i = 0; i < img.frusta.size(); ++i)
    {
        ASSERT_EQ(img.frusta[i].num_samples, loaded_img.frusta[i].num_samples);
        ASSERT_EQ(img.frusta[i].hist_idx, loaded_img.frusta[i].hist_idx);
    }

    for (size_t i = 0; i < img.frequencies.size(); ++i)
    {
        ASSERT_EQ(img.frequencies[i], loaded_img.frequencies[i]);
        ASSERT_EQ(img.bin_ids[i], loaded_img.bin_ids[i]);
    }
}

TEST(test_rayhistogram, test_get_bin)
{
    for (int b = 0; b < RayHistogramGenerationParameters::NUM_BINS; ++b)
    {
        float v = b / static_cast<float>(RayHistogramGenerationParameters::NUM_BINS - 1);
        auto bin = get_bin(v, 0.0f, 1.0f);
        ASSERT_EQ(bin, b);
    }
}

TEST(test_rayhistogram, test_simple_rayhistogram_sampler)
{
    SimpleRayHistogramSamplerHost sampler;

    vector<uint16_t> bins({1, RayHistogramGenerationParameters::NUM_BINS - 1, 42});
    vector<uint16_t> fs({10, 50, 40});

    vector<int> hist({0, 0, 0});

    for (int i = 0; i < 100; ++i)
    {
        auto s = sampler(bins.data(), fs.data(), 3, 100.f);
        ASSERT_LE(s, 1.f);
        ASSERT_GE(s, 0.0f);

        int bin = s * static_cast<float>(RayHistogramGenerationParameters::NUM_BINS);
        ASSERT_TRUE(bin == 1 || bin == (RayHistogramGenerationParameters::NUM_BINS - 1) || bin == 42);

        if (bin == 1)
            hist[0] += 1;
        else if (bin == 42)
            hist[1] += 1;
        else if (bin == RayHistogramGenerationParameters::NUM_BINS - 1)
            hist[2] += 1;
    }

    ASSERT_NEAR(hist[0], 10, 5);
    ASSERT_NEAR(hist[1], 50, 10);
    ASSERT_NEAR(hist[2], 40, 10);
}