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
#ifndef MOMENT_QUANTIZATION_H
#define MOMENT_QUANTIZATION_H

#include "common.h"
#include "moment_image.h"

namespace moment_quantization
{
extern bool uses_prediction_coding_quantization_table();

extern void load_prediction_coding_quantization_table(const string &file);

extern vector<Byte> get_prediction_coding_quantization_table(int max_moments, int numBits);
extern void set_prediction_coding_quantization_table(const vector<Byte> &table);
extern vector<Byte> get_prediction_coding_quantization_table();
extern void write_prediction_coding_quantization_table_to_file(const string &filename);

extern uint32_t quantize(float value, int numBits, float min, float max);
extern float dequantize(uint32_t q, int numBits, float min, float max);

extern vector<Byte> quantize_prediction_coding(const MomentImageHost &mi, const vector<Byte> &quant_table);
extern void dequantize_prediction_coding(MomentImageHost &mi, const vector<Byte> &quant_table, const vector<Byte> &qs);

extern vector<Byte> quantize_prediction_coding(const MomentImageHost &mi, int numBits);
extern void dequantize_prediction_coding(MomentImageHost &mi, int numBits, const vector<Byte> &qs);

extern vector<Byte> quantize(const MomentImageHost &mi, int numBits);
extern void dequantize(MomentImageHost &mi, int numBits, const vector<Byte> &qs);

// Old functions working on non compact images
extern vector<Byte> quantize(const vector<float> &moments, int numMoments, int numBits);
extern vector<float> dequantize(const vector<Byte> &qs, int numMoments, int numBits);
} // namespace moment_quantization

#endif // MOMENT_QUANTIZATION_H
