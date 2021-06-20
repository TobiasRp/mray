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
#include "measure_coding_error.h"
#include "moment_image.h"
#include "moment_image_io.h"
#include "moment_quantization.h"
#include "moment_prediction_coding.h"
#include "cut/timing.h"
#include "mese/MESE_dynamic.h"

#include <random>

#ifdef CUDA_SUPPORT
#include "measure_coding_error.cuh"
#endif

#include "H5Cpp.h"

inline float apply_quantization(float value, float b)
{
    assert(value >= 0.0f && value <= 1.0f); // Holds for prediction coding
    float pow = std::pow(2.0f, b);
    float q = std::floor(value * pow);
    return q / pow;
}

void apply_quantization(MomentImageHost &mi, int b_idx, float b)
{
#pragma omp parallel for default(none) shared(mi, b_idx, b)
    for (int y = 0; y < mi.height; ++y)
    {
        for (int x = 0; x < mi.width; ++x)
        {
            auto num_moments = mi.get_num_moments(x, y);
            if (num_moments <= b_idx)
                continue;

            auto idx = mi.get_idx(x, y);

            auto m_b = mi.data[idx + b_idx];
            mi.data[idx + b_idx] = apply_quantization(m_b, b);
        }
    }
}

void apply_quantization(MomentImageHost &mi, const vector<Byte> &table)
{
#pragma omp parallel for default(none) shared(mi, table)
    for (int y = 0; y < mi.height; ++y)
    {
        for (int x = 0; x < mi.width; ++x)
        {
            auto num_moments = mi.get_num_moments(x, y);
            if (num_moments == 0)
                continue;

            auto idx = mi.get_idx(x, y);

            for (int l = 0; l < num_moments; ++l)
            {
                auto m_b = mi.data[idx + l];
                mi.data[idx + l] = apply_quantization(m_b, table[l]);
            }
        }
    }
}



uint32_t compute_reference(const MomentImageHost &img, vector<float> &moments, vector<uint32_t> &indices)
{
    indices.resize(img.width * img.height, 0);

    uint32_t current_idx = 0;
    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            auto num_moments = img.get_num_moments(x, y);
            if (num_moments <= 0)
                continue;

            indices[y * img.width + x] = current_idx;
            ++current_idx;
        }
    }

    moments.resize(img.data.size());

    vector<float> params = img.get_coding_params();
#pragma omp parallel for default(none), shared(img, moments, params)
    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            auto num_moments = img.get_num_moments(x, y);

            if (num_moments <= 0)
                continue;

            auto idx = img.get_idx(x, y);
            vector<float_complex> ref_exp_moments(num_moments);
            vector<float_complex> ref_eval_poly(num_moments);

            vector<float_complex> temp_code(num_moments);
            vector<float_complex> temp(num_moments);

            transform_dequantization_real(num_moments, &img.data[idx], temp_code.data(),
                                          WarpParameters{img.coding_warp, params.data()});
            decode(num_moments, temp_code.data(), ref_exp_moments.data(), ref_eval_poly.data(), temp.data());

            exponentialToTrigonometricMoments(num_moments, temp.data(), ref_exp_moments.data());

            for (int l = 0; l < num_moments; ++l)
                moments[idx + l] = temp[l].x;
        }
    }
    return current_idx;
}

void compute_coding_error(const MomentImageHost &img, const vector<float> &ref_moments,
                          const vector<uint32_t> &indices, vector<float> &errors)
{
    vector<float> params = img.get_coding_params();
#pragma omp parallel for default(none), shared(img, ref_moments, indices, errors, params)
    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            auto num_moments = img.get_num_moments(x, y);

            if (num_moments <= 0)
                continue;

            auto idx = img.get_idx(x, y);
            vector<float_complex> q_exp_moments(num_moments);
            vector<float_complex> q_eval_poly(num_moments);

            vector<float_complex> temp_code(num_moments);
            vector<float_complex> temp(num_moments);

            transform_dequantization_real(num_moments, &img.data[idx], temp_code.data(),
                                          WarpParameters{img.coding_warp, params.data()});
            decode(num_moments, temp_code.data(), q_exp_moments.data(), q_eval_poly.data(), temp.data());

            exponentialToTrigonometricMoments(num_moments, temp.data(), q_exp_moments.data());

            auto sum = 0.0f;
            for (int l = 0; l < num_moments; ++l)
            {
                auto diff = ref_moments[idx+l] - temp[l].x;
                sum += diff * diff;
            }

            auto rmse = sqrtf(sum);
            // rRMSE: Divide by average
            errors[indices[y * img.width + x]] = rmse / ref_moments[idx];
        }
    }
}

void get_mean(const vector<float> &values, float &mean)
{
    mean = 0.0f;
    for (auto e : values)
    {
        mean += e;
    }
    mean /= values.size();
}

void write_error_stats(const string &output, float exp_mean_error, float num_bits, int idx, string name)
{
    H5::H5File file(output, H5F_ACC_RDWR);

    auto grp = file.createGroup(name);

    H5::DataType ftype(H5::PredType::NATIVE_FLOAT);

    hsize_t attr_dims[] = {1};
    auto attr_space = H5::DataSpace(1, attr_dims);

    H5::DataType itype(H5::PredType::NATIVE_INT32);
    if (num_bits > 0)
    {
        auto num_bits_attr = grp.createAttribute("num_bits", ftype, attr_space);
        num_bits_attr.write(ftype, &num_bits);
    }

    if (idx >= 0)
    {
        auto idx_attr = grp.createAttribute("idx", itype, attr_space);
        idx_attr.write(itype, &idx);
    }

    auto exp_mean_attr = grp.createAttribute("rRMSE", ftype, attr_space);
    exp_mean_attr.write(ftype, &exp_mean_error);
}

void write_table(const string &output, const vector<Byte> &table, string grp_name)
{
    H5::H5File file(output, H5F_ACC_RDWR);

    auto grp = file.openGroup(grp_name);

    hsize_t dims[] = {table.size()};
    H5::DataSpace data_space(1, dims);

    H5::DataType btype(H5::PredType::NATIVE_UINT8);
    auto dset = grp.createDataSet("table", btype, data_space);
    dset.write(table.data(), btype);
}

std::mt19937 gen;

void sample_table(const vector<int> &minima, const vector<int> &maxima, vector<Byte> &table)
{
    std::uniform_int_distribution<int> dist(1, table.size() - 1);
    int b_idx = dist(gen);

    std::uniform_int_distribution<int> bdist(minima[b_idx], maxima[b_idx]);
    auto bits = bdist(gen);
    if (bits == table[b_idx] && table[b_idx - 1] >= minima[b_idx])
        bits -= 1;
    else if (bits == table[b_idx] && table[b_idx + 1] <= maxima[b_idx])
        bits += 1;

    if (bits > table[b_idx])
    {
        table[b_idx] += 1;
        for (int b = b_idx - 1; b >= 0; --b)
        {
            if (table[b] < table[b_idx])
                table[b] = table[b_idx];
        }
    }
    else if (bits < table[b_idx])
    {
        table[b_idx] -= 1;
        for (int b = b_idx; b < static_cast<int>(table.size()); ++b)
        {
            if (table[b] > table[b_idx])
                table[b] = table[b_idx];
        }
    }
}

void sample_coding_quantization_curves(const string &moment_file, const string &output, const string &initial_curve)
{
    if (!isFile(output.c_str()))
        H5::H5File file(output, H5F_ACC_TRUNC);

    auto curve_len = initial_curve.length();
    string curve_id = initial_curve.substr(curve_len - 6, curve_len - 4);

    moment_quantization::load_prediction_coding_quantization_table(initial_curve);
    auto initial_curve_table = moment_quantization::get_prediction_coding_quantization_table();

    auto img = load_moment_image(moment_file);

    if (!img.prediction_code)
        img.prediction_encode(CODING_WARP_DEFAULT_TRANSFORMED, 1e-4f);

    vector<int> minima, maxima;
    for (int i = 0; i < img.num_moments; ++i)
    {
        minima.push_back(2);
        maxima.push_back(16);
    }

    vector<Byte> table(img.num_moments);
    std::copy(initial_curve_table.begin(), initial_curve_table.begin() + table.size(), table.begin());

#ifdef CUDA_SUPPORT
    cut::dev_ptr<uint32_t> d_indices(img.width * img.height);
    cut::dev_ptr<float> d_ref_moments(img.data.size());

    MomentImageDevice d_img(img);
    d_img.load_from(img);
    setup_coding_error_device(d_img);

    auto size = compute_reference_device(d_img, d_ref_moments.get(), d_indices.get());

    cut::dev_ptr<float> errors(size);
    cut::dev_ptr<float> orig_data(img.data.size());
    orig_data.loadFromDevice(d_img.data.get(), img.data.size());
#else
    vector<float> ref_moments;
    vector<uint32_t> indices;
    auto size = compute_reference(img, ref_moments, indices);
#endif

    size_t N = 2000;

    for (size_t id = 0; id < N; ++id)
    {
        // Reset once in a while
        if (id % 2 == 100)
            std::copy(initial_curve_table.begin(), initial_curve_table.begin() + table.size(), table.begin());

        sample_table(minima, maxima, table);

        float exp_error_mean;
#ifdef CUDA_SUPPORT
        apply_quantization_device(d_img, table);
        compute_coding_error_device(d_img, d_ref_moments.get(), errors.get(), d_indices.get(), size, exp_error_mean);
        d_img.data.loadFromDevice(orig_data.get(), img.data.size());
#else
        MomentImageHost mi_quantized(img);
        apply_quantization(mi_quantized, table);

        vector<float> exp_error(size);
        compute_coding_error(mi_quantized, ref_moments, indices, exp_error);
        get_mean(exp_error, exp_error_mean);
#endif

        int b = 0;
        for (int t : table)
            b += t;

        auto grp_name = curve_id + "_" + std::to_string(id);
        write_error_stats(output, exp_error_mean, b, id, grp_name);
        write_table(output, table, grp_name);
    }
}

#ifdef CUDA_SUPPORT
float measure_idx_error_device(const MomentImageDevice &d_img, const float *d_orig_data, const float *d_ref_moments,
                               float *d_errors, const uint32_t *d_indices, size_t size, int idx, int b)
{
    float exp_error_mean;
    apply_quantization_device(d_img, d_orig_data, idx, b);
    compute_coding_error_device(d_img, d_ref_moments, d_errors, d_indices, size, exp_error_mean);
    return exp_error_mean;
}
#endif

float measure_idx_error(const MomentImageHost &img, const vector<float> &ref,
                        const vector<uint32_t> &indices, size_t size, int idx, int b)
{
    MomentImageHost mi_quantized(img);
    apply_quantization(mi_quantized, idx, b);

    vector<float> errors(size);
    compute_coding_error(mi_quantized, ref, indices, errors);

    float error_mean;
    get_mean(errors, error_mean);
    return error_mean;
}

void create_quantization_table(const MomentImageHost &img, int start_bits)
{
    assert(img.prediction_code);

#ifdef CUDA_SUPPORT
    SCOPED_CUDA_QUERY("Quantization curve");

    cut::dev_ptr<uint32_t> d_indices(img.width * img.height);
    cut::dev_ptr<float> d_ref_moments(img.data.size());

    MomentImageDevice d_img(img);
    d_img.load_from(img);
    setup_coding_error_device(d_img);

    auto size = compute_reference_device(d_img, d_ref_moments.get(), d_indices.get());

    cut::dev_ptr<float> errors(size);
    cut::dev_ptr<float> orig_data(img.data.size());
    orig_data.loadFromDevice(d_img.data.get(), img.data.size());
#else
    SCOPED_CPU_QUERY("Quantization curve");

    vector<float> ref_moments;
    vector<uint32_t> indices;
    auto size = compute_reference(img, ref_moments, indices);
#endif

    vector<Byte> table(img.num_moments);
    table[0] = 16;
    table[1] = start_bits;

    int minimum_bits = 2;

#ifdef CUDA_SUPPORT
    float error_threshold = measure_idx_error_device(d_img, orig_data.get(), d_ref_moments.get(), errors.get(),
                                                     d_indices.get(), size, 1, start_bits);
    revert_quantization_device(d_img, 1, orig_data.get());
#else
    float error_threshold = measure_idx_error(img, ref_moments, indices, size, 1, start_bits);
#endif

    for (size_t idx = 2; idx < table.size(); ++idx)
    {
        table[idx] = table[idx - 1];
        for (int b = table[idx - 1]; b >= minimum_bits; --b)
        {
#ifdef CUDA_SUPPORT
            float exp_error = measure_idx_error_device(d_img, orig_data.get(), d_ref_moments.get(), errors.get(),
                                                       d_indices.get(), size, idx, b);
#else
            float exp_error = measure_idx_error(img, ref_moments, indices, size, idx, b);
#endif
            if (exp_error >= error_threshold)
                break;

            table[idx] = b;
        }

#ifdef CUDA_SUPPORT
        revert_quantization_device(d_img, idx, orig_data.get());
#endif
    }

    moment_quantization::set_prediction_coding_quantization_table(table);
}