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
#ifndef MRAY_PARAMETERS_H
#define MRAY_PARAMETERS_H

#include "common.h"
#include "cut/camera.h"

template <typename Parameters> FUNC float map_density(float s, const Parameters &p)
{
    return cut::clamp((s - p.data_min) / (p.data_max - p.data_min), 0.0f, 1.0f);
}

inline Vec3f parse_background_color(std::unordered_map<string, string> &params)
{
    // Default is white
    Vec3f color(1.f, 1.f, 1.f);
    if (params.find("background_r") != params.end())
        color.x = std::stof(params["background_r"]);
    if (params.find("background_g") != params.end())
        color.y = std::stof(params["background_g"]);
    if (params.find("background_b") != params.end())
        color.z = std::stof(params["background_b"]);
    return color;
}

struct GenerationParameters
{
    float step_size = 1.f;
    float step_size_write_samples = 1.5f;

    float data_min = 0.0f;
    float data_max = 1.0f;

    float error_threshold = 1.f;
    float bias = 1e-4f;

    bool compact_image = true;
    bool prediction_coding = true;
    bool compress = false;
    bool entropy_coding = false;
    Byte quantization_bits = 32;
    int coding_warp = 0;
    int quantization_curve_sampling = 64;

    bool transform_to_bounds = false;
    float transform_bound_eps = 0.01f;
};

inline GenerationParameters parse_gen_params(std::unordered_map<string, string> &params)
{
    GenerationParameters p;

    if (params.find("step_size") != params.end())
        p.step_size = std::stof(params["step_size"]);

    if (params.find("step_size_write_samples") != params.end())
        p.step_size_write_samples = std::stof(params["step_size_write_samples"]);

    if (params.find("data_min") != params.end())
        p.data_min = std::stof(params["data_min"]);

    if (params.find("data_max") != params.end())
        p.data_max = std::stof(params["data_max"]);

    if (params.find("error_threshold") != params.end())
        p.error_threshold = std::stof(params["error_threshold"]);

    if (params.find("bias") != params.end())
        p.bias = std::stof(params["bias"]);

    if (params.find("compact_image") != params.end())
        p.compact_image = std::stoi(params["compact_image"]) != 0;

    if (params.find("prediction_coding") != params.end())
        p.prediction_coding = std::stoi(params["prediction_coding"]) != 0;

    if (params.find("compress") != params.end())
        p.compress = std::stoi(params["compress"]) != 0;

    if (params.find("entropy_coding") != params.end())
        p.entropy_coding = std::stoi(params["entropy_coding"]) != 0;

    if (params.find("quantization_bits") != params.end())
        p.quantization_bits = std::stoi(params["quantization_bits"]);

    if (params.find("coding_warp") != params.end())
        p.coding_warp = std::stoi(params["coding_warp"]);

    if (params.find("quantization_curve_sampling") != params.end())
        p.quantization_curve_sampling = std::stoi(params["quantization_curve_sampling"]);

    if (params.find("transform_to_bounds") != params.end())
        p.transform_to_bounds = std::stoi(params["transform_to_bounds"]);

    if (params.find("transform_bound_eps") != params.end())
        p.transform_bound_eps = std::stof(params["transform_bound_eps"]);

    return p;
}

struct ErrorReconstructionParameters
{
    int error_type = ERROR_PERCENTILE;
    int error_percentile = 90;

    float bias = 1e-6f;
    float data_min = 0.0f;
    float data_max = 1.0f;
    float step_size = 0.01f;

    bool compress = false;
    bool entropy_coding = false;
    Byte quantization_bits = 32;
};

inline ErrorReconstructionParameters parse_error_params(std::unordered_map<string, string> &params)
{
    ErrorReconstructionParameters p;

    if (params.find("error_type") != params.end())
        p.error_type = std::stoi(params["error_type"]);

    if (params.find("error_percentile") != params.end())
        p.error_percentile = std::stoi(params["error_percentile"]);

    if (params.find("bias") != params.end())
        p.bias = std::stof(params["bias"]);

    if (params.find("data_min") != params.end())
        p.data_min = std::stof(params["data_min"]);

    if (params.find("data_max") != params.end())
        p.data_max = std::stof(params["data_max"]);

    if (params.find("step_size") != params.end())
        p.step_size = std::stof(params["step_size"]);

    if (params.find("compress") != params.end())
        p.compress = std::stoi(params["compress"]) != 0;

    if (params.find("entropy_coding") != params.end())
        p.entropy_coding = std::stoi(params["entropy_coding"]) != 0;

    if (params.find("quantization_bits") != params.end())
        p.quantization_bits = std::stoi(params["quantization_bits"]);

    return p;
}

struct SingleScatteringParameters
{
    bool enable = false;
    bool use_cache = true;

    int cache_size_x = 128;
    int cache_size_y = 128;
    int cache_size_z = 128;

    float step_size_factor = 1.0f;

    float henyey_greenstein_g = 0.0f;

    float direction_x = 0.0f;
    float direction_y = -1.0f;
    float direction_z = 0.1f;

    float intensity_r = 1.f;
    float intensity_g = 1.f;
    float intensity_b = 1.f;
};

inline SingleScatteringParameters parse_ss_params(std::unordered_map<string, string> &params)
{
    SingleScatteringParameters ss;
    if (params.find("ss_enable") != params.end())
        ss.enable = std::stoi(params["ss_enable"]);
    if (params.find("ss_use_cache") != params.end())
        ss.use_cache = std::stoi(params["ss_use_cache"]);
    if (params.find("ss_cache_size_x") != params.end())
        ss.cache_size_x = std::stoi(params["ss_cache_size_x"]);
    if (params.find("ss_cache_size_t") != params.end())
        ss.cache_size_y = std::stoi(params["ss_cache_size_y"]);
    if (params.find("ss_cache_size_z") != params.end())
        ss.cache_size_z = std::stoi(params["ss_cache_size_z"]);

    if (params.find("ss_step_size_factor") != params.end())
        ss.step_size_factor = std::stof(params["ss_step_size_factor"]);
    if (params.find("ss_henyey_greenstein_g") != params.end())
        ss.henyey_greenstein_g = std::stof(params["ss_henyey_greenstein_g"]);
    if (params.find("ss_intensity_r") != params.end())
        ss.intensity_r = std::stof(params["ss_intensity_r"]);
    if (params.find("ss_intensity_g") != params.end())
        ss.intensity_g = std::stof(params["ss_intensity_g"]);
    if (params.find("ss_intensity_b") != params.end())
        ss.intensity_b = std::stof(params["ss_intensity_b"]);

    if (params.find("light") != params.end())
    {
        auto light_cam = cut::load_from_file(params["light"], 1, 1);
        ss.direction_x = light_cam.dir.x;
        ss.direction_y = light_cam.dir.y;
        ss.direction_z = light_cam.dir.z;
    }

    return ss;
}

struct ReconstructionParameters
{
    float bias = 1e-6f;
    float medium_emission = 0.1f;
    float medium_scattering = 0.1f;
    float early_ray_termination = 0.95f;

    float data_min = 0.0f;
    float data_max = 1.0f;

    float step_size = 0.01f;
    float step_size_write_samples = 1.5f;

    bool use_truncated_fourier = false;

    SingleScatteringParameters ss;
};

inline ReconstructionParameters parse_rec_params(std::unordered_map<string, string> &params)
{
    ReconstructionParameters p;

    if (params.find("bias") != params.end())
        p.bias = std::stof(params["bias"]);

    if (params.find("medium_emission") != params.end())
        p.medium_emission = std::stof(params["medium_emission"]);

    if (params.find("medium_scattering") != params.end())
        p.medium_scattering = std::stof(params["medium_scattering"]);

    if (params.find("early_ray_termination") != params.end())
        p.early_ray_termination = std::stof(params["early_ray_termination"]);

    if (params.find("data_min") != params.end())
        p.data_min = std::stof(params["data_min"]);

    if (params.find("data_max") != params.end())
        p.data_max = std::stof(params["data_max"]);

    if (params.find("step_size") != params.end())
        p.step_size = std::stof(params["step_size"]);

    if (params.find("step_size_write_samples") != params.end())
        p.step_size_write_samples = std::stof(params["step_size_write_samples"]);

    if (params.find("use_truncated_fourier") != params.end())
        p.use_truncated_fourier = std::stoi(params["use_truncated_fourier"]);

    p.ss = parse_ss_params(params);

    return p;
}

struct ResamplingReconstructionParameters
{
    ReconstructionParameters rec;
    int width;
    int height;

    bool use_cache = false;
    int res_x = 128;
    int res_y = 64;
    int res_z = 64;
};

inline ResamplingReconstructionParameters parse_resampling_rec_params(std::unordered_map<string, string> &params)
{
    ResamplingReconstructionParameters p;
    p.rec = parse_rec_params(params);

    if (params.find("res_x") != params.end())
        p.res_x = std::stoi(params["res_x"]);
    if (params.find("res_y") != params.end())
        p.res_y = std::stoi(params["res_y"]);
    if (params.find("res_z") != params.end())
        p.res_z = std::stoi(params["res_z"]);
    if (params.find("res_width") != params.end())
        p.width = std::stoi(params["res_width"]);
    if (params.find("res_height") != params.end())
        p.height = std::stoi(params["res_height"]);
    if (params.find("res_use_cache") != params.end())
        p.use_cache = std::stoi(params["res_use_cache"]);

    return p;
}


struct UncertaintyReconstructionParameters
{
    ReconstructionParameters rec;

    float bound_interpolation = 0.5f;
    int error_type = RMSE;
};

inline UncertaintyReconstructionParameters parse_uncertainty_rec_params(std::unordered_map<string, string> &params)
{
    UncertaintyReconstructionParameters p;
    p.rec = parse_rec_params(params);

    if (params.find("bound_interpolation") != params.end())
        p.bound_interpolation = std::stof(params["bound_interpolation"]);
    if (params.find("error_type") != params.end())
        p.error_type = std::stoi(params["error_type"]);
    return p;
}

struct ReferenceParameters
{
    float step_size = 1.f;
    float medium_emission = 0.1f;
    float medium_scattering = 0.1f;
    float early_ray_termination = 0.95f;

    float data_min = 0.0f;
    float data_max = 1.0f;

    SingleScatteringParameters ss;
};

inline ReferenceParameters parse_ref_params(std::unordered_map<string, string> &params)
{
    ReferenceParameters p;

    if (params.find("step_size") != params.end())
        p.step_size = std::stof(params["step_size"]);

    if (params.find("medium_emission") != params.end())
        p.medium_emission = std::stof(params["medium_emission"]);

    if (params.find("medium_scattering") != params.end())
        p.medium_scattering = std::stof(params["medium_scattering"]);

    if (params.find("early_ray_termination") != params.end())
        p.early_ray_termination = std::stof(params["early_ray_termination"]);

    if (params.find("data_min") != params.end())
        p.data_min = std::stof(params["data_min"]);

    if (params.find("data_max") != params.end())
        p.data_max = std::stof(params["data_max"]);

    p.ss = parse_ss_params(params);

    return p;
}

inline ReferenceParameters ref_from_rec_params(const ReconstructionParameters &params)
{
    ReferenceParameters ref;
    ref.step_size = params.step_size;
    ref.medium_emission = params.medium_emission;
    ref.medium_scattering = params.medium_scattering;
    ref.early_ray_termination = params.early_ray_termination;
    ref.data_min = params.data_min;
    ref.data_max = params.data_max;
    ref.ss = params.ss;
    return ref;
}

struct RayHistogramGenerationParameters
{
    static constexpr int NUM_BINS = 256;

    float entropy_threshold = 0.1f;

    float step_size = 1.f;

    float data_min = 0.0f;
    float data_max = 1.0f;

    bool compress = false;
};

inline RayHistogramGenerationParameters parse_rayhist_gen_params(std::unordered_map<string, string> &params)
{
    RayHistogramGenerationParameters p;

    if (params.find("step_size") != params.end())
        p.step_size = std::stof(params["step_size"]);

    if (params.find("entropy_threshold") != params.end())
        p.entropy_threshold = std::stof(params["entropy_threshold"]);

    if (params.find("data_min") != params.end())
        p.data_min = std::stof(params["data_min"]);

    if (params.find("data_max") != params.end())
        p.data_max = std::stof(params["data_max"]);

    if (params.find("compress") != params.end())
        p.compress = std::stoi(params["compress"]) != 0;

    return p;
}

struct RayHistogramReconstructionParameters
{
    int num_samples = 512;

    float step_size = 1.f;
    float preintegration_step_size_factor = 1.f;
    float medium_emission = 0.1f;
    float early_ray_termination = 0.95f;

    float data_min = 0.0f;
    float data_max = 1.0f;
};

inline RayHistogramReconstructionParameters parse_rayhist_rec_params(std::unordered_map<string, string> &params)
{
    RayHistogramReconstructionParameters p;

    if (params.find("num_samples") != params.end())
        p.num_samples = std::stof(params["num_samples"]);

    if (params.find("step_size") != params.end())
        p.step_size = std::stof(params["step_size"]);

    if (params.find("medium_emission") != params.end())
        p.medium_emission = std::stof(params["medium_emission"]);

    if (params.find("early_ray_termination") != params.end())
        p.early_ray_termination = std::stof(params["early_ray_termination"]);

    if (params.find("data_min") != params.end())
        p.data_min = std::stof(params["data_min"]);

    if (params.find("data_max") != params.end())
        p.data_max = std::stof(params["data_max"]);

    return p;
}

#endif // MRAY_PARAMETERS_H