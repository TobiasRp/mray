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
#ifndef MRAY_MOMENT_RAYMARCHER_CUH
#define MRAY_MOMENT_RAYMARCHER_CUH

#include "volume.h"
#include "moment_image.h"
#include "color_image.h"
#include "samples_image.h"
#include "ray_histogram_image.h"
#include "single_scattering_image.h"
#include "transfer_function.h"
#include "cut/camera.h"
#include "parameters.h"

void generate_reference_device(cut::Camera cam, const Particles &volume, const TransferFunction &tf,
                               ColorImageHost &img, const ReferenceParameters &params);
void generate_reference_device(cut::Camera cam, const RegularGrid &volume, const TransferFunction &tf,
                               ColorImageHost &img, const ReferenceParameters &params);

void generate_samples_device(cut::Camera cam, const Particles &volume, SamplesImageHost &img,
                             const GenerationParameters &params);
void generate_samples_device(cut::Camera cam, const RegularGrid &volume, SamplesImageHost &img,
                             const GenerationParameters &params);

void generate_device(cut::Camera cam, const Particles &volume, MomentImageHost &img,
                     const GenerationParameters &params);
void generate_device(cut::Camera cam, const RegularGrid &volume, MomentImageHost &img,
                     const GenerationParameters &params);

void reconstruct_device(cut::Camera cam, const MomentImageHost &mi, const TransferFunction &tf, ColorImageHost &cimg,
                        const ReconstructionParameters &params);

void reconstruct_device(cut::Camera cam, const MomentImageHost &mi, const TransferFunction &tf, ColorImageHost &cimg,
                        const SingleScatteringImageHost &ss, const ReconstructionParameters &params);

void generate_reference_reconstructed_ss_device(cut::Camera cam, const TransferFunction &tf, const RegularGrid &volume,
                                                ColorImageHost &img, const SingleScatteringImageHost &ss,
                                                const ReconstructionParameters &params);
void generate_reference_reconstructed_ss_device(cut::Camera cam, const TransferFunction &tf, const Particles &volume,
                                                ColorImageHost &img, const SingleScatteringImageHost &ss,
                                                const ReconstructionParameters &params);

void reconstruct_resampled_device(cut::Camera cam, const MomentImageHost &mi, const TransferFunction &tf,
                                  ColorImageHost &cimg, const vector<cut::Camera> &new_cam, const string &output,
                                  Vec3f background, const ResamplingReconstructionParameters &params);

void reconstruct_samples_device(cut::Camera cam, const MomentImageHost &mi, SamplesImageHost &img,
                                const ReconstructionParameters &params);

void reconstruct_error_device(cut::Camera cam, const Particles &volume, MomentImageHost &mi,
                              const ErrorReconstructionParameters &params);
void reconstruct_error_device(cut::Camera cam, const RegularGrid &volume, MomentImageHost &mi,
                              const ErrorReconstructionParameters &params);

void reconstruct_uncertainty_interpolation_device(cut::Camera cam, const MomentImageHost &mi,
                                                  const TransferFunction &tf, ColorImageHost &cimg,
                                                  const UncertaintyReconstructionParameters &params);
void reconstruct_uncertainty_interpolation_device(cut::Camera cam, const MomentImageHost &mi,
                                                  const TransferFunction &tf, ColorImageHost &cimg,
                                                  const SingleScatteringImageHost &ss,
                                                  const UncertaintyReconstructionParameters &params);

void reconstruct_uncertainty_convolution_device(cut::Camera cam, const MomentImageHost &mi, const TransferFunction &tf,
                                                ColorImageHost &cimg,
                                                const UncertaintyReconstructionParameters &params);

void reconstruct_rayhistogram_device(cut::Camera cam, const RayHistogramImageHost &img, const TransferFunction &tf,
                                     ColorImageHost &cimg, const RayHistogramReconstructionParameters &params);

void reconstruct_rayhistogram_samples_device(cut::Camera cam, const RayHistogramImageHost &img, SamplesImageHost &simg,
                                             const RayHistogramReconstructionParameters &params);

#endif // MRAY_MOMENT_RAYMARCHER_CUH
