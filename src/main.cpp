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
#include <iostream>

#include "cut/camera.h"
#include "cut/timing.h"
#include "cut/timing_log_writer.h"
#include "regular_grid_loader.h"
#include "particle_loader.h"
#include "moment_raymarcher.h"
#include "moment_image_interpolator.h"
#include "moment_image_io.h"
#include "moment_image_sample.h"

#include "measure_coding_error.h"
#include "moment_quantization.h"

void parse_quantization_curve_param(unordered_map<string, string> &params)
{
    if (params.find("quantization_table") != params.end())
        moment_quantization::load_prediction_coding_quantization_table(params["quantization_table"]);
}

void init_logging(unordered_map<string, string> &params)
{
    if (params.find("logfile") != params.end())
    {
        string logfile = params["logfile"];

        string log_grp = "default";
        if (params.find("log_grp") != params.end())
            log_grp = params["log_grp"];

        TimeLogWriter::init(logfile, log_grp);
    }
    else
        TimeLogWriter::init_no_logging();
}

unordered_map<string, string> parse_parameters(string params)
{
    vector<string> pairs;

    auto start = 0U;
    auto end = params.find(',');

    while (end != std::string::npos)
    {
        pairs.push_back(params.substr(start, end - start));

        start = end + 1;
        end = params.find(",", start);
    }

    pairs.push_back(params.substr(start, end - start));

    unordered_map<string, string> parameters;
    for (auto pstr : pairs)
    {
        auto end = pstr.find('=');

        string name = pstr.substr(0, end);
        auto valueStr = pstr.substr(end + 1, pstr.length());
        parameters[name] = valueStr;
    }

    return parameters;
}

inline float get_preintegration_step_size(unordered_map<string, string> &params)
{
    float preintegration_step_size_factor = 1.f;
    if (params.find("preintegration_step_size_factor") != params.end())
        preintegration_step_size_factor = std::stof(params["preintegration_step_size_factor"]);
    return preintegration_step_size_factor;
}

inline TransferFunction get_tf(unordered_map<string, string> &params, float step_size)
{
    string tf_file = params["tf"];
    float preintegration_step_size_factor = get_preintegration_step_size(params);
    return read_from_disk(tf_file.c_str(), step_size * preintegration_step_size_factor);
}

cut::Vec2i parse_upsampling(unordered_map<string, string> &params)
{
    cut::Vec2i factor(1, 1);
    if (params.find("upsample_x") != params.end())
        factor.x = std::stoi(params["upsample_x"]);
    if (params.find("upsample_y") != params.end())
        factor.y = std::stoi(params["upsample_y"]);
    return factor;
}

inline MomentImageHost get_moment_img(string moment_image_file, unordered_map<string, string> params)
{
    MomentImageHost mi = load_moment_image(moment_image_file);
    auto upsampling_factor = parse_upsampling(params);
    if (upsampling_factor.x > 1 || upsampling_factor.y > 1)
        mi = MomentImageInterpolator::upsample(mi, upsampling_factor);
    return mi;
}

inline SingleScatteringImageHost get_ss_img(unordered_map<string, string> params)
{
    assert(params.find("ss_img_file") != params.end());
    auto ss_img = load_moment_image(params["ss_img_file"]);
    auto light_cam = cut::load_from_file(params["light"], ss_img.width, ss_img.height);
    return SingleScatteringImageHost(light_cam, ss_img);
}

void reference(string output_file, string dataset, unordered_map<string, string> params)
{
    init_logging(params);
    auto ref_params = parse_ref_params(params);

    auto tf = get_tf(params, ref_params.step_size);

    int t = std::stoi(params["t"]);
    string var = params["variable"];

    auto upsampling_factor = parse_upsampling(params);
    int width = std::stoi(params["width"]) * upsampling_factor.x;
    int height = std::stoi(params["height"]) * upsampling_factor.y;

    string camera_file = params["camera"];
    auto cam = cut::load_from_file(camera_file, width, height);

    ColorImageHost img(width, height);
    if (dataset.substr(dataset.size() - 3, dataset.size()) == ".nc")
    {
        RegularGridLoader loader(dataset);
        auto volume = loader.get(t, var);
        generate_reference(cam, tf, volume, img, ref_params);
    }
    else
    {
        float smoothing_length = std::stof(params["smoothing_length"]);

        ParticleLoader loader(dataset);
        auto volume = loader.get(t, var, smoothing_length);
        generate_reference(cam, tf, volume, img, ref_params);
    }

    write_PPM(output_file, img, parse_background_color(params));
}

void reference_reference_reconstructed_ss(string output_file, string dataset, unordered_map<string, string> params)
{
    init_logging(params);
    parse_quantization_curve_param(params);

    ReconstructionParameters rec_params = parse_rec_params(params);
    if (!rec_params.ss.enable)
        return;

    int t = std::stoi(params["t"]);
    string var = params["variable"];

    auto tf = get_tf(params, rec_params.step_size);
    string camera_file = params["camera"];

    auto upsampling_factor = parse_upsampling(params);
    int width = std::stoi(params["width"]) * upsampling_factor.x;
    int height = std::stoi(params["height"]) * upsampling_factor.y;

    auto cam = cut::load_from_file(camera_file, width, height);

    ColorImageHost img(width, height);
    if (dataset.substr(dataset.size() - 3, dataset.size()) == ".nc")
    {
        RegularGridLoader loader(dataset);
        auto volume = loader.get(t, var);
        generate_reference_reconstructed_ss(cam, tf, volume, img, get_ss_img(params), rec_params);
    }
    else
    {
        float smoothing_length = std::stof(params["smoothing_length"]);

        ParticleLoader loader(dataset);
        auto volume = loader.get(t, var, smoothing_length);
        generate_reference_reconstructed_ss(cam, tf, volume, img, get_ss_img(params), rec_params);
    }

    write_PPM(output_file, img, parse_background_color(params));
}

void generate_samples(string output_file, string dataset, unordered_map<string, string> params)
{
    init_logging(params);

    int t = std::stoi(params["t"]);
    string var = params["variable"];
    int width = std::stoi(params["width"]);
    int height = std::stoi(params["height"]);

    auto gen_params = parse_gen_params(params);

    string camera_file = params["camera"];
    auto cam = cut::load_from_file(camera_file, width, height);

    SamplesImageHost img(width, height);
    if (dataset.substr(dataset.size() - 3, dataset.size()) == ".nc")
    {
        RegularGridLoader loader(dataset);
        auto volume = loader.get(t, var);
        generate_samples(cam, volume, img, gen_params);
    }
    else
    {
        float smoothing_length = std::stof(params["smoothing_length"]);

        ParticleLoader loader(dataset);
        auto volume = loader.get(t, var, smoothing_length);
        generate_samples(cam, volume, img, gen_params);
    }

    write_samples_image(output_file, img);
}

void generate_rayhistogram(string output_file, string dataset, unordered_map<string, string> params)
{
    init_logging(params);

    int t = std::stoi(params["t"]);
    string var = params["variable"];
    int width = std::stoi(params["width"]);
    int height = std::stoi(params["height"]);

    auto gen_params = parse_rayhist_gen_params(params);

    string camera_file = params["camera"];
    auto cam = cut::load_from_file(camera_file, width, height);

    RayHistogramImageHost img(width, height);
    if (dataset.substr(dataset.size() - 3, dataset.size()) == ".nc")
    {
        RegularGridLoader loader(dataset);
        auto volume = loader.get(t, var);
        img.domain = volume.domain;
        generate_rayhistogram(cam, volume, img, gen_params);
    }
    else
    {
        float smoothing_length = std::stof(params["smoothing_length"]);

        ParticleLoader loader(dataset);
        auto volume = loader.get(t, var, smoothing_length);
        img.domain = volume.domain;
        generate_rayhistogram(cam, volume, img, gen_params);
    }

    write_rayhistogram_image(output_file, img, gen_params.compress);
}

void generate(string moment_image_file, string dataset, unordered_map<string, string> params)
{
    init_logging(params);
    parse_quantization_curve_param(params);

    int t = std::stoi(params["t"]);
    string var = params["variable"];
    int width = std::stoi(params["width"]);
    int height = std::stoi(params["height"]);
    int num_moments = std::stoi(params["num_moments"]);
    if (num_moments > 255)
    {
        // We assume num_moments <= 255 for moment image io. We think this is a reasonable limitation.
        printf("Number of moments cannot be greater than 255! This is an implementation limitation!\n");
        std::terminate();
    }

    auto gen_params = parse_gen_params(params);

    string camera_file = params["camera"];
    auto cam = cut::load_from_file(camera_file, width, height);

    MomentImageHost mi(width, height, num_moments, gen_params.compact_image, gen_params.transform_to_bounds);
    if (dataset.substr(dataset.size() - 3, dataset.size()) == ".nc")
    {
        RegularGridLoader loader(dataset);
        auto volume = loader.get(t, var);
        mi.domain = volume.domain;
        generate(cam, volume, mi, gen_params);
    }
    else
    {
        float smoothing_length = std::stof(params["smoothing_length"]);

        ParticleLoader loader(dataset);
        auto volume = loader.get(t, var, smoothing_length);
        mi.domain = volume.domain;
        generate(cam, volume, mi, gen_params);
    }

    if (gen_params.prediction_coding && gen_params.quantization_bits < 32)
    {
        int sw = cut::min(mi.width, gen_params.quantization_curve_sampling);
        int sh = cut::min(mi.height, gen_params.quantization_curve_sampling);
        create_quantization_table(sample_moment_image(mi, sw, sh), gen_params.quantization_bits);

        if (params.find("write_table") != params.end())
            moment_quantization::write_prediction_coding_quantization_table_to_file(params["write_table"]);

    }

    write_moment_image(mi, moment_image_file, gen_params.compress, gen_params.quantization_bits,
                       gen_params.entropy_coding);
}

void compute_errors(string moment_image_file, string dataset, unordered_map<string, string> params)
{
    init_logging(params);
    parse_quantization_curve_param(params);

    ErrorReconstructionParameters error_params = parse_error_params(params);

    auto mi = get_moment_img(moment_image_file, params);
    mi.add_error_bounds();

    int t = std::stoi(params["t"]);
    string var = params["variable"];

    string camera_file = params["camera"];
    auto cam = cut::load_from_file(camera_file, mi.width, mi.height);

    if (dataset.substr(dataset.size() - 3, dataset.size()) == ".nc")
    {
        RegularGridLoader loader(dataset);
        auto volume = loader.get(t, var);
        mi.domain = volume.domain;
        reconstruct_error(cam, volume, mi, error_params);
    }
    else
    {
        float smoothing_length = std::stof(params["smoothing_length"]);

        ParticleLoader loader(dataset);
        auto volume = loader.get(t, var, smoothing_length);
        mi.domain = volume.domain;
        reconstruct_error(cam, volume, mi, error_params);
    }

    write_moment_image(mi, moment_image_file, error_params.compress, error_params.quantization_bits,
                       error_params.entropy_coding);
}

void reconstruct(string moment_image_file, string output_file, unordered_map<string, string> params)
{
    init_logging(params);
    parse_quantization_curve_param(params);

    ReconstructionParameters rec_params = parse_rec_params(params);

    auto tf = get_tf(params, rec_params.step_size);

    auto mi = get_moment_img(moment_image_file, params);

    string camera_file = params["camera"];
    auto cam = cut::load_from_file(camera_file, mi.width, mi.height);

    ColorImageHost img(mi.width, mi.height);
    if (rec_params.ss.enable)
        reconstruct(cam, mi, tf, img, get_ss_img(params), rec_params);
    else
        reconstruct(cam, mi, tf, img, rec_params);

    write_PPM(output_file, img, parse_background_color(params));
}

void reconstruct_resampled(string moment_image_file, string output_file, unordered_map<string, string> params)
{
    init_logging(params);
    parse_quantization_curve_param(params);

    auto res_params = parse_resampling_rec_params(params);

    auto tf = get_tf(params, res_params.rec.step_size);

    auto mi = get_moment_img(moment_image_file, params);

    string camera_file = params["camera"];
    auto cam = cut::load_from_file(camera_file, mi.width, mi.height);

    auto cam_path = params["res_camera_path"];
    auto cam_first = std::stoi(params["res_camera_first"]);
    auto cam_num = std::stoi(params["res_camera_num"]);
    vector<cut::Camera> new_cams;
    for (int i = cam_first; i < cam_first + cam_num; ++i)
        new_cams.push_back(cut::load_from_file(cam_path + "camera_" + std::to_string(i) + ".txt", res_params.width,
                                               res_params.height));

    ColorImageHost img(res_params.width, res_params.height);
    reconstruct_resampled(cam, mi, tf, img, new_cams, output_file, parse_background_color(params), res_params);
}

void reconstruct_time_series(string moment_image_file_start, string moment_image_file_end, string output_prefix,
                             unordered_map<string, string> params)
{
    init_logging(params);
    parse_quantization_curve_param(params);

    ReconstructionParameters rec_params = parse_rec_params(params);

    int num_time_steps = 2;
    if (params.find("time_steps") != params.end())
        num_time_steps = std::stoi(params["time_steps"]);

    auto start = get_moment_img(moment_image_file_start, params);
    auto end = get_moment_img(moment_image_file_end, params);

    string camera_file = params["camera"];
    auto cam = cut::load_from_file(camera_file, start.width, start.height);

    MomentImageInterpolator interp(start, end);
    for (int t = 0; t < num_time_steps; ++t)
    {
        string tf_file_1 = params["tf1"];
        string tf_file_2 = params["tf2"];
        float preintegration_step_size_factor = get_preintegration_step_size(params);
        auto tf =
            read_from_disk_interpolate(tf_file_1.c_str(), tf_file_2.c_str(), t / static_cast<float>(num_time_steps),
                                       rec_params.step_size * preintegration_step_size_factor);

        auto mi = interp.get(t / static_cast<float>(num_time_steps));

        ColorImageHost img(mi.width, mi.height);
        reconstruct(cam, mi, tf, img, rec_params);

        write_PPM(output_prefix + std::to_string(t) + ".ppm", img, parse_background_color(params));
    }
}

void reconstruct_samples(string moment_image_file, string output_file, unordered_map<string, string> params)
{
    init_logging(params);
    parse_quantization_curve_param(params);

    string camera_file = params["camera"];

    ReconstructionParameters rec_params = parse_rec_params(params);

    auto mi = get_moment_img(moment_image_file, params);

    auto cam = cut::load_from_file(camera_file, mi.width, mi.height);

    SamplesImageHost img(mi.width, mi.height);
    reconstruct_samples(cam, mi, img, rec_params);

    write_samples_image(output_file, img);
}

void reconstruct_uncertainty_interpolate(string moment_image_file, string output_file,
                                         unordered_map<string, string> params)
{
    init_logging(params);
    parse_quantization_curve_param(params);

    UncertaintyReconstructionParameters uncertainty_params = parse_uncertainty_rec_params(params);

    auto tf = get_tf(params, uncertainty_params.rec.step_size);

    auto mi = get_moment_img(moment_image_file, params);

    string camera_file = params["camera"];
    auto cam = cut::load_from_file(camera_file, mi.width, mi.height);

    ColorImageHost img(mi.width, mi.height);
    if (uncertainty_params.rec.ss.enable)
        reconstruct_uncertainty_interpolation(cam, mi, tf, img, get_ss_img(params), uncertainty_params);
    else
        reconstruct_uncertainty_interpolation(cam, mi, tf, img, uncertainty_params);

    write_PPM(output_file, img, parse_background_color(params));
}

void reconstruct_uncertainty_convolution(string moment_image_file, string output_file,
                                         unordered_map<string, string> params)
{
    init_logging(params);
    parse_quantization_curve_param(params);

    UncertaintyReconstructionParameters uncertainty_params = parse_uncertainty_rec_params(params);

    float preintegration_step_size_factor = get_preintegration_step_size(params);
    auto tf = read_from_disk_uncertainty(params["tf"].c_str(),
                                         uncertainty_params.rec.step_size * preintegration_step_size_factor,
                                         uncertainty_params.error_type);

    auto mi = get_moment_img(moment_image_file, params);

    string camera_file = params["camera"];
    auto cam = cut::load_from_file(camera_file, mi.width, mi.height);

    ColorImageHost img(mi.width, mi.height);
    reconstruct_uncertainty_convolution(cam, mi, tf, img, uncertainty_params);

    write_PPM(output_file, img, parse_background_color(params));
}

void reconstruct_rayhist(string rayhist_image_file, string output_file, unordered_map<string, string> params)
{
    init_logging(params);

    RayHistogramReconstructionParameters rec_params = parse_rayhist_rec_params(params);

    string tf_file = params["tf"];
    float preintegration_step_size_factor = get_preintegration_step_size(params);
    rec_params.step_size *= preintegration_step_size_factor;
    auto tf = read_from_disk_1D(tf_file.c_str());

    auto img = read_rayhistogram_image(rayhist_image_file);

    string camera_file = params["camera"];
    auto cam = cut::load_from_file(camera_file, img.width, img.height);

    ColorImageHost cimg(img.width, img.height);
    reconstruct_rayhistogram(cam, img, tf, cimg, rec_params);

    write_PPM(output_file, cimg, parse_background_color(params));
}

void reconstruct_rayhist_samples(string rayhist_image_file, string output_file, unordered_map<string, string> params)
{
    init_logging(params);

    string camera_file = params["camera"];

    RayHistogramReconstructionParameters rec_params = parse_rayhist_rec_params(params);

    auto img = read_rayhistogram_image(rayhist_image_file);

    auto cam = cut::load_from_file(camera_file, img.width, img.height);

    SamplesImageHost simg(img.width, img.height);
    reconstruct_rayhistogram_samples(cam, img, simg, rec_params);

    write_samples_image(output_file, simg);
}

int main(int argc, char *argv[])
{
#ifdef CUDA_SUPPORT
    init_device(0);
#endif

    string mode(argv[1]);

    if (mode == "--generate")
    {
        if (argc != 5)
        {
            std::cout << "Invalid arguments!" << std::endl;
            std::terminate();
        }

        auto moment_image = string(argv[2]);
        auto dataset = string(argv[3]);
        auto params = string(argv[4]);

        generate(moment_image, dataset, parse_parameters(params));
    }
    else if (mode == "--error-bounds")
    {
        if (argc != 5)
        {
            std::cout << "Invalid arguments!" << std::endl;
            std::terminate();
        }

        auto moment_image = string(argv[2]);
        auto dataset = string(argv[3]);
        auto params = string(argv[4]);

        compute_errors(moment_image, dataset, parse_parameters(params));
    }
    else if (mode == "--generate-samples")
    {
        if (argc != 5)
        {
            std::cout << "Invalid arguments!" << std::endl;
            std::terminate();
        }

        auto output_image = string(argv[2]);
        auto dataset = string(argv[3]);
        auto params = string(argv[4]);

        generate_samples(output_image, dataset, parse_parameters(params));
    }
    else if (mode == "--generate-rayhist")
    {
        if (argc != 5)
        {
            std::cout << "Invalid arguments!" << std::endl;
            std::terminate();
        }

        auto output_image = string(argv[2]);
        auto dataset = string(argv[3]);
        auto params = string(argv[4]);

        generate_rayhistogram(output_image, dataset, parse_parameters(params));
    }
    else if (mode == "--reconstruct")
    {
        if (argc != 5)
        {
            std::cout << "Invalid arguments!" << std::endl;
            std::terminate();
        }

        auto moment_image = argv[2];
        auto output_file = argv[3];
        auto params = argv[4];

        reconstruct(moment_image, output_file, parse_parameters(params));
    }
    else if (mode == "--reconstruct-samples")
    {
        if (argc != 5)
        {
            std::cout << "Invalid arguments!" << std::endl;
            std::terminate();
        }

        auto moment_image = argv[2];
        auto output_file = argv[3];
        auto params = argv[4];

        reconstruct_samples(moment_image, output_file, parse_parameters(params));
    }
    else if (mode == "--reconstruct-resampled")
    {
        if (argc != 5)
        {
            std::cout << "Invalid arguments!" << std::endl;
            std::terminate();
        }

        auto moment_image = argv[2];
        auto output_file = argv[3];
        auto params = argv[4];

        reconstruct_resampled(moment_image, output_file, parse_parameters(params));
    }
    else if (mode == "--reconstruct-time")
    {
        if (argc != 6)
        {
            std::cout << "Invalid arguments!" << std::endl;
            std::terminate();
        }

        auto moment_image_start = argv[2];
        auto moment_image_end = argv[3];
        auto output_prefix = argv[4];
        auto params = argv[5];

        reconstruct_time_series(moment_image_start, moment_image_end, output_prefix, parse_parameters(params));
    }
    else if (mode == "--reconstruct-rayhist")
    {
        if (argc != 5)
        {
            std::cout << "Invalid arguments!" << std::endl;
            std::terminate();
        }

        auto rayhist_image = argv[2];
        auto output_file = argv[3];
        auto params = argv[4];

        reconstruct_rayhist(rayhist_image, output_file, parse_parameters(params));
    }
    else if (mode == "--reconstruct-rayhist-samples")
    {
        if (argc != 5)
        {
            std::cout << "Invalid arguments!" << std::endl;
            std::terminate();
        }

        auto rayhist_image = argv[2];
        auto output_file = argv[3];
        auto params = argv[4];

        reconstruct_rayhist_samples(rayhist_image, output_file, parse_parameters(params));
    }
    else if (mode == "--uncertainty-interpolate")
    {
        if (argc != 5)
        {
            std::cout << "Invalid arguments!" << std::endl;
            std::terminate();
        }

        auto moment_image = argv[2];
        auto output_file = argv[3];
        auto params = argv[4];

        reconstruct_uncertainty_interpolate(moment_image, output_file, parse_parameters(params));
    }
    else if (mode == "--uncertainty-convolution")
    {
        if (argc != 5)
        {
            std::cout << "Invalid arguments!" << std::endl;
            std::terminate();
        }

        auto moment_image = argv[2];
        auto output_file = argv[3];
        auto params = argv[4];

        reconstruct_uncertainty_convolution(moment_image, output_file, parse_parameters(params));
    }
    else if (mode == "--reference")
    {
        if (argc != 5)
        {
            std::cout << "Invalid arguments!" << std::endl;
            std::terminate();
        }

        auto output_file = argv[2];
        auto dataset = argv[3];
        auto params = argv[4];

        reference(output_file, dataset, parse_parameters(params));
    }
    else if (mode == "--reference-reconstructed-ss")
    {
        if (argc != 5)
        {
            std::cout << "Invalid arguments!" << std::endl;
            std::terminate();
        }

        auto output_file = argv[2];
        auto dataset = argv[3];
        auto params = argv[4];

        reference_reference_reconstructed_ss(output_file, dataset, parse_parameters(params));
    }
    else if (mode == "--sample-coding-quantization-curves")
    {
        sample_coding_quantization_curves(argv[2], argv[3], argv[4]);
    }
    else
    {
        std::cout << "Invalid arguments!" << std::endl;
        std::terminate();
    }

    g_TimeLogger.syncAndReport();
    TimeLogWriter::destroy();

#ifdef CUDA_SUPPORT
    cut::reportUnreleasedDeviceMemory();
#endif

    return 0;
}
