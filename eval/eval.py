import json
import glob
import argparse
import numpy as np
import collections.abc

from mray_cli import *
from eval_parameters import EvalParameters, dict_to_arguments

cfg_file = "config.json"


def process(cfg, eval_def):
    """ Processes, that is executes, an evaluation definition file. """

    dataset = eval_def['dataset']
    assert (os.path.isfile(dataset))

    params = EvalParameters(eval_def)

    frames = eval_def['frames']

    log(cfg, "Generating image files...")

    clear_output_dir(eval_def)

    for k, v in frames.items():
        params.set_frame_args(v)

        if params.compute_reference():
            reference_image_file = os.path.join(eval_def['output'], k + '_reference.ppm')
            reference_frame(cfg, dataset, params.get_reference_params(), reference_image_file)

            convert_ppm_to_png(
                os.path.join(os.path.join(eval_def['output'], k + '_reference')))

        if params.compute_generate():
            moment_image_file = os.path.join(eval_def['output'], k + '_moments.bin')
            generate_frame(cfg, dataset, params.get_generate_params(), moment_image_file)

            if 'generate_single-scattering' in eval_def and 'light' in v:
                ss_image_file = os.path.join(eval_def['output'], k + '_moments_ss.bin')
                generate_frame(cfg, dataset, params.get_generate_ss_params(v), ss_image_file)
                params.set_ss_image_file(ss_image_file)

        if params.compute_reference_reconstructed_ss():
            reference_reconstructed_ss_file = os.path.join(eval_def['output'], k + '_reference_rec_ss')
            frame_reference_reconstructed_ss(cfg, dataset, params.get_reference_reconstructed_ss_params(),
                                             reference_reconstructed_ss_file + '.ppm')
            convert_ppm_to_png(reference_reconstructed_ss_file)

        if params.compute_errors():
            moment_image_file = os.path.join(eval_def['output'], k + '_moments.bin')
            compute_error_frame(cfg, dataset, params.get_error_params(), moment_image_file)

        if params.compute_reconstruct():
            reconstructed_file = os.path.join(eval_def['output'], k + '_reconstructed.ppm')
            reconstruct_frame(cfg, params.get_reconstruct_params(), moment_image_file, reconstructed_file)

            convert_ppm_to_png(
                os.path.join(os.path.join(eval_def['output'], k + '_reconstructed')))

        if params.compute_reconstruction_fourier():
            reconstructed_fourier_file = os.path.join(eval_def['output'], k + '_reconstructed_fourier.ppm')
            reconstruct_frame(cfg, params.get_reconstruction_fourier_params(), moment_image_file,
                              reconstructed_fourier_file)

            convert_ppm_to_png(
                os.path.join(os.path.join(eval_def['output'], k + '_reconstructed_fourier')))

            reconstructed_fourier_samples_file = os.path.join(eval_def['output'],
                                                              k + '_reconstructed_fourier_samples.bin')
            reconstruct_samples_frame(cfg, params.get_reconstruction_fourier_params(), moment_image_file,
                                      reconstructed_fourier_samples_file)

        if params.compute_generate_samples():
            samples_image_file = os.path.join(eval_def['output'], k + '_generate_samples.bin')
            generate_samples_frame(cfg, dataset, params.get_generate_samples_params(), samples_image_file)

        if params.compute_reconstruct_samples():
            reconstructed_samples_file = os.path.join(eval_def['output'], k + '_reconstructed_samples.bin')
            reconstruct_samples_frame(cfg, params.get_reconstruct_samples_params(), moment_image_file,
                                      reconstructed_samples_file)

        if params.compute_rayhist_generate():
            rayhist_file = os.path.join(eval_def['output'], k + '_rayhist.bin')
            rayhist_generate_frame(cfg, dataset, params.get_rayhist_generate_params(), rayhist_file)

        if params.compute_rayhist_reconstruct():
            rayhist_reconstructed_file = os.path.join(eval_def['output'], k + '_rayhist.ppm')
            rayhist_reconstruct_frame(cfg, rayhist_file, params.get_rayhist_reconstruct_params(),
                                      rayhist_reconstructed_file)

            convert_ppm_to_png(os.path.join(os.path.join(eval_def['output'], k + '_rayhist')))

        if params.compute_rayhist_reconstruct_samples():
            rayhist_reconstructed_samples_file = os.path.join(eval_def['output'], k + '_rayhist_samples.bin')
            rayhist_reconstruct_samples_frame(cfg, rayhist_file, params.get_rayhist_reconstruct_samples_params(),
                                              rayhist_reconstructed_samples_file)

        if params.compute_reconstruction_resampled():
            reconstructed_resampled_file = os.path.join(eval_def['output'], k + '_reconstructed_resampled')
            reconstruct_resampled_frame(cfg, moment_image_file, params.get_reconstruction_resampled_params(),
                                        reconstructed_resampled_file)
            for f in glob.glob(reconstructed_resampled_file + '*.ppm'):
                convert_ppm_to_png(os.path.splitext(f)[0])
                os.remove(f)

        if params.compute_uncertainty_interpolation():
            num_interpolations = params.get_uncertainty_interpolation_num()

            for i in range(num_interpolations):
                file = os.path.join(eval_def['output'], k + '_uncertainty_interpolated' + '_' + str(i))
                uncertainty_interpolation_file = file + '.ppm'
                uncertainty_interpolation_frame(cfg, params.get_uncertainty_interpolation_params(i), moment_image_file,
                                                uncertainty_interpolation_file)
                convert_ppm_to_png(file)
                os.remove(file + '.ppm')

        log(cfg, "Finished frame: \"" + k + "\"!")

        if cfg['single_frame']:
            break


def load_cfg():
    global cfg_file
    if not os.path.isfile(cfg_file):
        cfg_file = os.path.join("eval/", cfg_file)

        if not os.path.isfile(cfg_file):
            print("Can't find config.json file! Start from project directory!")
            exit()

    with open(cfg_file, "r") as config_file:
        cfg = json.load(config_file)

    return cfg


def load_eval_def(filename):
    with open(filename, "r") as file:
        defs = json.load(file)

        if 'parent_file' in defs:
            parent_defs = load_eval_def(defs['parent_file'])

            # Overwrite parent entries
            for k, v in defs.items():
                if k not in parent_defs or not isinstance(v, collections.abc.Mapping):
                    parent_defs[k] = v
                else:
                    # Don't completely overwrite dicts...
                    for r_k, r_v in v.items():
                        parent_defs[k][r_k] = r_v
            return parent_defs
        else:
            return defs


class EvalArgs:
    def __init__(self, eval_def, use_cuda=True, single_frame=False, artifacts=None):
        self.use_cuda = use_cuda
        self.single_frame = single_frame
        self.eval_def = eval_def
        self.artifacts = artifacts


def run_eval(arguments):
    cfg = load_cfg()

    cfg['use_cuda'] = arguments.use_cuda
    cfg['single_frame'] = arguments.single_frame

    try:
        compile_mray(cfg)
    except:
        print("Could not compile project. Continuing without...")

    if isinstance(arguments.eval_def, collections.abc.Mapping):
        # Already loaded!
        defs = arguments.eval_def
    else:
        eval_def_file = arguments.eval_def

        if not os.path.isfile(eval_def_file):
            print(eval_def_file, 'is not a valid file! Exiting...')
            exit()

        defs = load_eval_def(eval_def_file)

    if hasattr(arguments, 'artifacts') and arguments.artifacts:
        defs['artifacts'] = arguments.artifacts

    process(cfg, defs)

    return defs['output']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mray evaluation')
    parser.add_argument('eval_def', type=str, help='Evaluation definition file')
    parser.add_argument('--single-frame', action='store_true', dest='single_frame',
                        help='Processes only the first frame')
    parser.add_argument('--cuda', action='store_true', dest='use_cuda', help='Enables CUDA')
    parser.add_argument('--no-cuda', action='store_false', dest='use_cuda', help='Disables CUDA')
    args = parser.parse_args()

    run_eval(args)

