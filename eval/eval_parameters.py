import os


def dict_to_arguments(params):
    """ Creates a comma separated argument string from a dict. """
    if len(params) == 0:
        return ''

    arguments = ''
    for k, v in params.items():
        arguments += k + '=' + str(v) + ','
    return arguments[:-1]


def append_arguments(params_a, params_b):
    """ Concatenates two argument strings. """
    if len(params_a) == 0:
        return params_b
    elif len(params_b) == 0:
        return params_a
    else:
        return params_a + ',' + params_b


def append_arguments3(p_a, p_b, p_c):
    """ Concatenates three argument strings. """
    return append_arguments(append_arguments(p_a, p_b), p_c)


class EvalParameters:
    """ Provides simpler access to the eval. parameters in an evaluation config. """

    def __init__(self, eval_def, disable_ss=False):
        self.artifacts = eval_def['artifacts']
        self.common_params = ''
        if 'common' in eval_def:
            self.common_params = dict_to_arguments(eval_def['common'])

        if 'perf_logging' in eval_def and eval_def['perf_logging']:
            logfile_name = 'log.hdf5'
            self.common_params = append_arguments(self.common_params,
                                                  'logfile=' + os.path.join(eval_def['output'], logfile_name))
            self.write_log = True
        else:
            self.write_log = False

        self.ref_params = dict_to_arguments(eval_def['reference'])
        self.gen_params = dict_to_arguments(eval_def['generation'])
        self.rec_params = dict_to_arguments(eval_def['reconstruction'])

        self.reconstruction_fourier = 'reconstruction_fourier' in eval_def

        if 'rayhist_generation' in eval_def:
            self.gen_rayhist_params = dict_to_arguments(eval_def['rayhist_generation'])

        if 'rayhist_reconstruction' in eval_def:
            self.rec_rayhist_params = dict_to_arguments(eval_def['rayhist_reconstruction'])

        if 'reconstruction_resampled' in eval_def:
            self.rec_sampled = dict_to_arguments(eval_def['reconstruction_resampled'])

        if 'error_bounds' in eval_def:
            self.error_params = dict_to_arguments(eval_def['error_bounds'])
            self.error_params = append_arguments(self.gen_params, self.error_params)

        if 'uncertainty_interpolation' in eval_def:
            self.uncertainty_interpolation_values = eval_def['uncertainty_interpolation']['bound_interpolation']

        if 'generate_single-scattering' in eval_def and not disable_ss:
            self.gen_ss_params = dict_to_arguments(eval_def['generate_single-scattering'])

        if 'use_single-scattering' in eval_def and not disable_ss:
            self.use_ss_params = dict_to_arguments(eval_def['use_single-scattering'])
            self.common_params = append_arguments(self.common_params, self.use_ss_params)

        self.frame_args = ''

    def get_log_params(self, name):
        if self.write_log:
            return 'log_grp=' + name
        else:
            return ''

    def set_frame_args(self, frame):
        self.frame_args = append_arguments(dict_to_arguments(frame), self.common_params)

    def append_frame_args(self, param):
        self.frame_args = append_arguments(self.frame_args, param)

    def set_ss_image_file(self, ss_img_file):
        if self.use_ss_params != '':
            self.frame_args = append_arguments(self.frame_args, 'ss_img_file=' + ss_img_file)

    def compute_reference(self):
        return 'all' in self.artifacts or 'reference' in self.artifacts or 'comparison' in self.artifacts

    def get_reference_params(self):
        return append_arguments3(self.frame_args, self.ref_params, self.get_log_params('reference'))

    def compute_reference_reconstructed_ss(self):
        return 'ref_rec_ss' in self.artifacts

    def get_reference_reconstructed_ss_params(self):
        return append_arguments3(self.frame_args, self.ref_params, self.rec_params)

    def compute_generate_samples(self):
        return 'all' in self.artifacts or 'generate-samples' in self.artifacts or 'comparison' in self.artifacts

    def get_generate_samples_params(self):
        return append_arguments(self.frame_args, self.gen_params)

    def compute_generate(self):
        return 'all' in self.artifacts or 'generate' in self.artifacts or 'reconstruct' in self.artifacts or 'comparison' in self.artifacts

    def get_generate_params(self):
        return append_arguments3(self.frame_args, self.gen_params, self.get_log_params('generate'))

    def get_generate_ss_params(self, v):
        ss_v = v.copy()
        ss_v['camera'] = v['light']
        frame_args = append_arguments(dict_to_arguments(ss_v), self.common_params)
        return append_arguments3(frame_args, self.gen_ss_params, self.get_log_params('generate-ss'))

    def compute_errors(self):
        return hasattr(self, 'error_params') and ('all' in self.artifacts or 'error-bounds' in self.artifacts)

    def get_error_params(self):
        return append_arguments3(self.frame_args, self.error_params, self.get_log_params('error-bounds'))

    def compute_reconstruct(self):
        return 'all' in self.artifacts or 'reconstruct' in self.artifacts or 'comparison' in self.artifacts

    def get_reconstruct_params(self):
        return append_arguments3(self.frame_args, self.rec_params, self.get_log_params('reconstruction'))

    def compute_reconstruct_samples(self):
        return 'all' in self.artifacts or 'reconstruct-samples' in self.artifacts or 'comparison' in self.artifacts

    def get_reconstruct_samples_params(self):
        return append_arguments(self.frame_args, self.rec_params)

    def compute_reconstruction_fourier(self):
        return self.reconstruction_fourier and (
                'all' in self.artifacts or 'comparison' in self.artifacts or 'fourier' in self.artifacts)

    def get_reconstruction_fourier_params(self):
        return append_arguments3(self.frame_args, self.rec_params,
                                 append_arguments(self.get_log_params('Fourier reconstruction'),
                                                  'use_truncated_fourier=1'))
                                                  
    def compute_reconstruction_resampled(self):
        return hasattr(self, "rec_sampled") and (
                "all" in self.artifacts or 'resampled' in self.artifacts)

    def get_reconstruction_resampled_params(self):
        return append_arguments3(self.frame_args, self.rec_sampled, self.rec_params)

    def compute_uncertainty_interpolation(self):
        return hasattr(self,
                       'uncertainty_interpolation_values') and 'all' in self.artifacts or 'uncertainty-interpolation' in self.artifacts

    def get_uncertainty_interpolation_num(self):
        return len(self.uncertainty_interpolation_values)

    def get_uncertainty_interpolation_params(self, i):
        return append_arguments(self.frame_args,
                                append_arguments3(self.rec_params, "bound_interpolation=" + str(
                                    self.uncertainty_interpolation_values[i]),
                                                  self.get_log_params('uncertainty-interpolate')))

    def compute_rayhist_generate(self):
        return hasattr(self, "gen_rayhist_params") and (
                'all' in self.artifacts or 'rayhistogram' in self.artifacts or 'comparison' in self.artifacts)

    def get_rayhist_generate_params(self):
        return append_arguments3(self.frame_args, self.gen_rayhist_params,
                                 self.get_log_params('generate-ray-histogram'))

    def compute_rayhist_reconstruct(self):
        return hasattr(self, "rec_rayhist_params") and (
                'all' in self.artifacts or 'rayhistogram' in self.artifacts or 'comparison' in self.artifacts)

    def get_rayhist_reconstruct_params(self):
        return append_arguments3(self.frame_args, self.rec_rayhist_params,
                                 self.get_log_params('reconstruct-ray-histogram'))

    def compute_rayhist_reconstruct_samples(self):
        return hasattr(self, "rec_rayhist_params") and (
                'all' in self.artifacts or 'rayhistogram' in self.artifacts or 'comparison' in self.artifacts)

    def get_rayhist_reconstruct_samples_params(self):
        return append_arguments(self.frame_args, self.rec_rayhist_params)
