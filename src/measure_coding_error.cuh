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
#ifndef MRAY_MEASURE_CODING_ERROR_CUH
#define MRAY_MEASURE_CODING_ERROR_CUH

#include "common.h"
#include "moment_image_device.cuh"

extern void setup_coding_error_device(const MomentImageDevice &d_img);

extern void apply_quantization_device(const MomentImageDevice &d_img, int b_idx, float b);
extern void apply_quantization_device(const MomentImageDevice &d_img, const float *d_orig_data, int b_idx, float b);
extern void apply_quantization_device(const MomentImageDevice &d_img, const vector<Byte> &table);

extern void revert_quantization_device(const MomentImageDevice &d_img, int b_idx, const float *d_orig_data);

/**
 * Compute bounded trigonometric moments as a reference for subsequent error measurements.
 */
extern uint32_t compute_reference_device(const MomentImageDevice &img, float *d_moments, uint32_t *d_indices);

/**
 * Compute the moments of the given moment image and compute an error to the given reference.
 */
extern void compute_coding_error_device(const MomentImageDevice &img, const float *d_ref_moments, float *d_errors,
                                        const uint32_t *d_indices, size_t size, float &error);

#endif // MRAY_MEASURE_CODING_ERROR_CUH
