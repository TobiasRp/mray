import subprocess
import os
from PIL import Image


def compile_mray(cfg):
    """ Compiles mray"""
    use_cuda = 'ON' if cfg['use_cuda'] else 'OFF'
    subprocess.run(['cmake', '-DENABLE_CUDA=' + use_cuda, '-DCMAKE_BUILD_TYPE=Release', '..', ], check=True,
                   cwd=cfg['mray_path'], stdout=subprocess.DEVNULL)
    subprocess.run([cfg['build_cmd']], check=True, cwd=cfg['mray_path'], stdout=subprocess.DEVNULL)

    if cfg['logging']:
        print('Finished Compiling!')


def mray_bin(cfg):
    return cfg['mray_path'] + '/mray'


def log(cfg, msg, *arguments):
    if cfg['logging']:
        print(msg, *arguments)


def log_params(cfg, *arguments):
    if cfg['log_params']:
        print('\t', *arguments)
        
        
def convert_ppm_to_png(file_no_ext):
    """ Converts a ppm from mray to a png."""
    img = Image.open(file_no_ext + '.ppm')
    img.save(file_no_ext + '.png')


def clear_output_dir(eval_def):
    """ Removes all existing files in the output directory. """
    if not os.path.isdir(eval_def['output']):
        os.mkdir(eval_def['output'])
    for f in os.listdir(eval_def['output']):
        os.remove(os.path.join(eval_def['output'], f))


def reference_frame(cfg, dataset, params, output_file):
    """ Create a reference image without analysis """
    log_params(cfg, '--reference', output_file, dataset, params)
    subprocess.run([mray_bin(cfg), '--reference', output_file, dataset, params], check=True,
                   stdout=subprocess.DEVNULL)


def frame_reference_reconstructed_ss(cfg, dataset, params, output_file):
    """ Create a reference image without analysis """
    log_params(cfg, '--reference-reconstructed-ss', output_file, dataset, params)
    subprocess.run([mray_bin(cfg), '--reference-reconstructed-ss', output_file, dataset, params], check=True,
                   stdout=subprocess.DEVNULL)


def generate_samples_frame(cfg, dataset, params, output_file):
    """ Generates a moment image for a single frame. """
    log_params(cfg, '--generate-samples', output_file, dataset, params)
    subprocess.run([mray_bin(cfg), '--generate-samples', output_file, dataset, params], check=True,
                   stdout=subprocess.DEVNULL)


def generate_frame(cfg, dataset, params, moment_image_file):
    """ Generates a moment image for a single frame. """
    log_params(cfg, '--generate', moment_image_file, dataset, params)
    subprocess.run([mray_bin(cfg), '--generate', moment_image_file, dataset, params], check=True,
                   stdout=subprocess.DEVNULL)


def compute_error_frame(cfg, dataset, params, moment_image_file):
    """ Computes and adds error bounds to a moment image for a single frame. """
    log_params(cfg, '--error-bounds', moment_image_file, dataset, params)
    subprocess.run([mray_bin(cfg), '--error-bounds', moment_image_file, dataset, params], check=True,
                   stdout=subprocess.DEVNULL)


def reconstruct_frame(cfg, params, moment_image_file, reconstructed_file):
    """ Reconstructs a moment image for a single frame. """
    log_params(cfg, '--reconstruct', moment_image_file, reconstructed_file, params)
    subprocess.run([mray_bin(cfg), '--reconstruct', moment_image_file, reconstructed_file, params], check=True,
                   stdout=subprocess.DEVNULL)


def reconstruct_frame_time_series(cfg, params, image_file_start, image_file_end, reconstructed_prefix):
    """ Reconstructs a moment image for a single frame. """
    log_params(cfg, '--reconstruct-time', image_file_start, image_file_end, reconstructed_prefix, params)
    subprocess.run(
        [mray_bin(cfg), '--reconstruct-time', image_file_start, image_file_end, reconstructed_prefix, params],
        check=True,
        stdout=subprocess.DEVNULL)


def reconstruct_samples_frame(cfg, params, moment_image_file, output_file):
    """ Generates a moment image for a single frame. """
    log_params(cfg, '--reconstruct-samples', moment_image_file, output_file, params)
    subprocess.run([mray_bin(cfg), '--reconstruct-samples', moment_image_file, output_file, params], check=True,
                   stdout=subprocess.DEVNULL)


def uncertainty_interpolation_frame(cfg, params, moment_image_file, output_file):
    log_params(cfg, '--uncertainty-interpolate', moment_image_file, output_file, params)
    subprocess.run([mray_bin(cfg), '--uncertainty-interpolate', moment_image_file, output_file, params], check=True,
                   stdout=subprocess.DEVNULL)


def uncertainty_convolution_frame(cfg, params, moment_image_file, output_file):
    log_params(cfg, '--uncertainty-convolution', moment_image_file, output_file, params)
    subprocess.run([mray_bin(cfg), '--uncertainty-convolution', moment_image_file, output_file, params], check=True,
                   stdout=subprocess.DEVNULL)


def rayhist_generate_frame(cfg, dataset, params, rayhist_file):
    log_params(cfg, '--generate-rayhist', rayhist_file, dataset, params)
    subprocess.run([mray_bin(cfg), '--generate-rayhist', rayhist_file, dataset, params], check=True,
                   stdout=subprocess.DEVNULL)


def rayhist_reconstruct_frame(cfg, rayhist_file, params, output_file):
    log_params(cfg, '--reconstruct-rayhist', rayhist_file, output_file, params)
    subprocess.run([mray_bin(cfg), '--reconstruct-rayhist', rayhist_file, output_file, params], check=True,
                   stdout=subprocess.DEVNULL)


def rayhist_reconstruct_samples_frame(cfg, rayhist_file, params, output_file):
    log_params(cfg, '--reconstruct-rayhist-samples', rayhist_file, output_file, params)
    subprocess.run([mray_bin(cfg), '--reconstruct-rayhist-samples', rayhist_file, output_file, params], check=True,
                   stdout=subprocess.DEVNULL)


def reconstruct_resampled_frame(cfg, moment_image_file, params, output_file):
    log_params(cfg, '--reconstruct-resampled', moment_image_file, output_file, params)
    subprocess.run([mray_bin(cfg), '--reconstruct-resampled', moment_image_file, output_file, params], check=True,
                   stdout=subprocess.DEVNULL)
