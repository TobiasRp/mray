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
#include "moment_image_interpolator.h"

MomentImageInterpolator::MomentImageInterpolator(const MomentImageHost &start, const MomentImageHost &end)
    : m_start(start)
    , m_end(end)
    , m_template(start.width, start.height, start.num_moments, true, start.has_bounds())
{
    assert(start.width == end.width && start.height == end.height);

    m_template.domain = start.domain;

    if (start.has_error_bounds())
        m_template.add_error_bounds();

#pragma omp parallel for default(none) shared(m_template, start, end)
    for (int y = 0; y < start.height; ++y)
        for (int x = 0; x < start.width; ++x)
            m_template.index[y * start.width + x] = std::max(start.get_num_moments(x, y), end.get_num_moments(x, y));

    m_template.is_compact = false;
    m_template.compact();
}

MomentImageHost MomentImageInterpolator::get(float t)
{
    if (t == 0.0f)
        return m_start;
    else if (t == 1.0f)
        return m_end;

#pragma omp parallel for default(none) shared(m_template, m_start, m_end, t)
    for (int y = 0; y < m_template.height; ++y)
    {
        for (int x = 0; x < m_template.width; ++x)
        {
            auto template_idx = m_template.get_idx(x, y);

            auto max_moments = std::max(m_start.get_num_moments(x, y), m_end.get_num_moments(x, y));
            //assert(m_template.get_num_moments(x, y) == max_moments);

            vector<float> start_moments(max_moments, 0.0f);
            m_start.get_trig_moments(x, y, start_moments);

            vector<float> end_moments(max_moments, 0.0f);
            m_end.get_trig_moments(x, y, end_moments);

            for (int l = 0; l < max_moments; ++l)
            {
                m_template.data[template_idx + l] =
                    (1.f - t) * start_moments[l] + t * end_moments[l];
            }

            if (m_template.has_bounds())
            {
                auto b = (1.f - t) * m_start.get_bounds(x, y) + t * m_end.get_bounds(x, y);
                m_template.set_bounds(x, y, DensityBound(b.x, b.y));
            }

            if (m_template.has_error_bounds())
            {
                auto e = (1.f - t) * m_start.get_error_bound(x, y) + t * m_end.get_error_bound(x, y);
                m_template.set_error_bound(x, y, e);
            }
        }
    }
    return m_template;
}

MomentImageHost MomentImageInterpolator::upsample(const MomentImageHost &img, cut::Vec2i factor)
{
    MomentImageHost big(img.width * factor.x, img.height * factor.y, img.num_moments, true, img.has_bounds());
    if (img.has_error_bounds())
        big.add_error_bounds();

    big.domain = img.domain;

#pragma omp parallel for default(none) shared(img, big, factor)
    for (int y = 0; y < big.height; ++y)
    {
        for (int x = 0; x < big.width; ++x)
        {
            auto img_x = x / static_cast<float>(factor.x);
            auto img_y = y / static_cast<float>(factor.y);

            auto i_x = cut::clamp(static_cast<int>(img_x), 0, img.width - 1);
            auto i_y = cut::clamp(static_cast<int>(img_y), 0, img.height - 1);
            auto i_x_1 = cut::clamp(static_cast<int>(img_x) + 1, 0, img.width - 1);
            auto i_y_1 = cut::clamp(static_cast<int>(img_y) + 1, 0, img.height - 1);

            auto num_moments_x = std::max(img.get_num_moments(i_x, i_y), img.get_num_moments(i_x_1, i_y));
            auto num_moments_y = std::max(img.get_num_moments(i_x, i_y_1), img.get_num_moments(i_x_1, i_y_1));
            auto num_moments = std::max(num_moments_x, num_moments_y);

            if (num_moments == 0)
                continue;

            auto t_x = img_x - std::floor(img_x);
            auto t_y = img_y - std::floor(img_y);

            if (big.has_bounds())
            {
                Vec2f bounds = (1.f - t_x) * (1.f - t_y) * img.get_bounds(i_x, i_y) +
                               t_x * (1.f - t_y) * img.get_bounds(i_x_1, i_y) +
                               (1.f - t_x) * t_y * img.get_bounds(i_x, i_y_1) +
                               t_x * t_y * img.get_bounds(i_x_1, i_y_1);
                big.set_bounds(x, y, DensityBound(bounds.x, bounds.y));
            }

            if (big.has_error_bounds())
            {
                auto e = (1.f - t_x) * (1.f - t_y) * img.get_error_bound(i_x, i_y) +
                         t_x * (1.f - t_y) * img.get_error_bound(i_x_1, i_y) +
                         (1.f - t_x) * t_y * img.get_error_bound(i_x, i_y_1) +
                         t_x * t_y * img.get_error_bound(i_x_1, i_y_1);
                big.set_error_bound(x, y, e);
            }

            vector<float> bl(num_moments, 0.0f);
            vector<float> br(num_moments, 0.0f);
            vector<float> ul(num_moments, 0.0f);
            vector<float> ur(num_moments, 0.0f);

            img.get_trig_moments(i_x, i_y, bl);
            img.get_trig_moments(i_x_1, i_y, br);
            img.get_trig_moments(i_x, i_y_1, ul);
            img.get_trig_moments(i_x_1, i_y_1, ur);

            big.index[y * big.width + x] = num_moments;

            auto big_idx = big.get_idx(x, y);
            for (int l = 0; l < num_moments; ++l)
            {
                big.data[big_idx + l] = (1.f - t_x) * (1.f - t_y) * bl[l] + t_x * (1.f - t_y) * br[l] +
                                        (1.f - t_x) * t_y * ul[l] + t_x * t_y * ur[l];
            }
        }
    }
    big.index[big.width * big.height] = 0;

    big.compact();

    return big;
}
