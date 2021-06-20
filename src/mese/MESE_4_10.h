// Copyright (c) 2019, Christoph Peters
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

/**
 * Implementation of bounded MESE based on code from Christoph Peters.
 * Only supports m=4 and m=10 numbers of moments.
 * (Used for unit testing)
 */

#pragma once
#include <cmath>
#include "complex_algebra.h"


FUNC void trigonometricToExponentialMoments4(float_complex pOutExponentialMoment[4],const float pTrigonometricMoment[4]){
	float zerothMomentPhase=3.14159265f*pTrigonometricMoment[0]-1.57079633f;
	pOutExponentialMoment[0]=float_complex(cosf(zerothMomentPhase),sinf(zerothMomentPhase));
	pOutExponentialMoment[0]=0.0795774715f*pOutExponentialMoment[0];
	pOutExponentialMoment[1]=pTrigonometricMoment[1]*float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0];
	pOutExponentialMoment[2]=pTrigonometricMoment[2]*float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]+pTrigonometricMoment[1]*float_complex(0.0f,3.14159265f)*pOutExponentialMoment[1];
	pOutExponentialMoment[3]=pTrigonometricMoment[3]*float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]+pTrigonometricMoment[2]*float_complex(0.0f,4.1887902f)*pOutExponentialMoment[1]+pTrigonometricMoment[1]*float_complex(0.0f,2.0943951f)*pOutExponentialMoment[2];
	pOutExponentialMoment[0]=2.0f*pOutExponentialMoment[0];
}

FUNC void trigonometricToExponentialMoments10(float_complex pOutExponentialMoment[10],const float pTrigonometricMoment[10]){
	float zerothMomentPhase=3.14159265f*pTrigonometricMoment[0]-1.57079633f;
	pOutExponentialMoment[0]=float_complex(cosf(zerothMomentPhase),sinf(zerothMomentPhase));
	pOutExponentialMoment[0]=0.0795774715f*pOutExponentialMoment[0];
	pOutExponentialMoment[1]=pTrigonometricMoment[1]*float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0];
	pOutExponentialMoment[2]=pTrigonometricMoment[2]*float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]+pTrigonometricMoment[1]*float_complex(0.0f,3.14159265f)*pOutExponentialMoment[1];
	pOutExponentialMoment[3]=pTrigonometricMoment[3]*float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]+pTrigonometricMoment[2]*float_complex(0.0f,4.1887902f)*pOutExponentialMoment[1]+pTrigonometricMoment[1]*float_complex(0.0f,2.0943951f)*pOutExponentialMoment[2];
	pOutExponentialMoment[4]=pTrigonometricMoment[4]*float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]+pTrigonometricMoment[3]*float_complex(0.0f,4.71238898f)*pOutExponentialMoment[1]+pTrigonometricMoment[2]*float_complex(0.0f,3.14159265f)*pOutExponentialMoment[2]+pTrigonometricMoment[1]*float_complex(0.0f,1.57079633f)*pOutExponentialMoment[3];
	pOutExponentialMoment[5]=pTrigonometricMoment[5]*float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]+pTrigonometricMoment[4]*float_complex(0.0f,5.02654825f)*pOutExponentialMoment[1]+pTrigonometricMoment[3]*float_complex(0.0f,3.76991118f)*pOutExponentialMoment[2]+pTrigonometricMoment[2]*float_complex(0.0f,2.51327412f)*pOutExponentialMoment[3]+pTrigonometricMoment[1]*float_complex(0.0f,1.25663706f)*pOutExponentialMoment[4];
	pOutExponentialMoment[6]=pTrigonometricMoment[6]*float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]+pTrigonometricMoment[5]*float_complex(0.0f,5.23598776f)*pOutExponentialMoment[1]+pTrigonometricMoment[4]*float_complex(0.0f,4.1887902f)*pOutExponentialMoment[2]+pTrigonometricMoment[3]*float_complex(0.0f,3.14159265f)*pOutExponentialMoment[3]+pTrigonometricMoment[2]*float_complex(0.0f,2.0943951f)*pOutExponentialMoment[4]+pTrigonometricMoment[1]*float_complex(0.0f,1.04719755f)*pOutExponentialMoment[5];
	pOutExponentialMoment[7]=pTrigonometricMoment[7]*float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]+pTrigonometricMoment[6]*float_complex(0.0f,5.38558741f)*pOutExponentialMoment[1]+pTrigonometricMoment[5]*float_complex(0.0f,4.48798951f)*pOutExponentialMoment[2]+pTrigonometricMoment[4]*float_complex(0.0f,3.5903916f)*pOutExponentialMoment[3]+pTrigonometricMoment[3]*float_complex(0.0f,2.6927937f)*pOutExponentialMoment[4]+pTrigonometricMoment[2]*float_complex(0.0f,1.7951958f)*pOutExponentialMoment[5]+pTrigonometricMoment[1]*float_complex(0.0f,0.897597901f)*pOutExponentialMoment[6];
	pOutExponentialMoment[8]=pTrigonometricMoment[8]*float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]+pTrigonometricMoment[7]*float_complex(0.0f,5.49778714f)*pOutExponentialMoment[1]+pTrigonometricMoment[6]*float_complex(0.0f,4.71238898f)*pOutExponentialMoment[2]+pTrigonometricMoment[5]*float_complex(0.0f,3.92699082f)*pOutExponentialMoment[3]+pTrigonometricMoment[4]*float_complex(0.0f,3.14159265f)*pOutExponentialMoment[4]+pTrigonometricMoment[3]*float_complex(0.0f,2.35619449f)*pOutExponentialMoment[5]+pTrigonometricMoment[2]*float_complex(0.0f,1.57079633f)*pOutExponentialMoment[6]+pTrigonometricMoment[1]*float_complex(0.0f,0.785398163f)*pOutExponentialMoment[7];
	pOutExponentialMoment[9]=pTrigonometricMoment[9]*float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]+pTrigonometricMoment[8]*float_complex(0.0f,5.58505361f)*pOutExponentialMoment[1]+pTrigonometricMoment[7]*float_complex(0.0f,4.88692191f)*pOutExponentialMoment[2]+pTrigonometricMoment[6]*float_complex(0.0f,4.1887902f)*pOutExponentialMoment[3]+pTrigonometricMoment[5]*float_complex(0.0f,3.4906585f)*pOutExponentialMoment[4]+pTrigonometricMoment[4]*float_complex(0.0f,2.7925268f)*pOutExponentialMoment[5]+pTrigonometricMoment[3]*float_complex(0.0f,2.0943951f)*pOutExponentialMoment[6]+pTrigonometricMoment[2]*float_complex(0.0f,1.3962634f)*pOutExponentialMoment[7]+pTrigonometricMoment[1]*float_complex(0.0f,0.698131701f)*pOutExponentialMoment[8];
	pOutExponentialMoment[0]=2.0f*pOutExponentialMoment[0];
}

FUNC void trigonometricToExponentialMoments4(float_complex pOutExponentialMoment[4],const float_complex pTrigonometricMoment[4]){
	float zerothMomentPhase=3.14159265f*pTrigonometricMoment[0].x-1.57079633f;
	pOutExponentialMoment[0]=float_complex(cosf(zerothMomentPhase),sinf(zerothMomentPhase));
	pOutExponentialMoment[0]=0.0795774715f*pOutExponentialMoment[0];
	pOutExponentialMoment[1]=float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]*pTrigonometricMoment[1];
	pOutExponentialMoment[2]=float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]*pTrigonometricMoment[2]+float_complex(0.0f,3.14159265f)*pOutExponentialMoment[1]*pTrigonometricMoment[1];
	pOutExponentialMoment[3]=float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]*pTrigonometricMoment[3]+float_complex(0.0f,4.1887902f)*pOutExponentialMoment[1]*pTrigonometricMoment[2]+float_complex(0.0f,2.0943951f)*pOutExponentialMoment[2]*pTrigonometricMoment[1];
	pOutExponentialMoment[0]=2.0f*pOutExponentialMoment[0];
}

FUNC void trigonometricToExponentialMoments10(float_complex pOutExponentialMoment[10],const float_complex pTrigonometricMoment[10]){
	float zerothMomentPhase=3.14159265f*pTrigonometricMoment[0].x-1.57079633f;
	pOutExponentialMoment[0]=float_complex(cosf(zerothMomentPhase),sinf(zerothMomentPhase));
	pOutExponentialMoment[0]=0.0795774715f*pOutExponentialMoment[0];
	pOutExponentialMoment[1]=float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]*pTrigonometricMoment[1];
	pOutExponentialMoment[2]=float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]*pTrigonometricMoment[2]+float_complex(0.0f,3.14159265f)*pOutExponentialMoment[1]*pTrigonometricMoment[1];
	pOutExponentialMoment[3]=float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]*pTrigonometricMoment[3]+float_complex(0.0f,4.1887902f)*pOutExponentialMoment[1]*pTrigonometricMoment[2]+float_complex(0.0f,2.0943951f)*pOutExponentialMoment[2]*pTrigonometricMoment[1];
	pOutExponentialMoment[4]=float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]*pTrigonometricMoment[4]+float_complex(0.0f,4.71238898f)*pOutExponentialMoment[1]*pTrigonometricMoment[3]+float_complex(0.0f,3.14159265f)*pOutExponentialMoment[2]*pTrigonometricMoment[2]+float_complex(0.0f,1.57079633f)*pOutExponentialMoment[3]*pTrigonometricMoment[1];
	pOutExponentialMoment[5]=float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]*pTrigonometricMoment[5]+float_complex(0.0f,5.02654825f)*pOutExponentialMoment[1]*pTrigonometricMoment[4]+float_complex(0.0f,3.76991118f)*pOutExponentialMoment[2]*pTrigonometricMoment[3]+float_complex(0.0f,2.51327412f)*pOutExponentialMoment[3]*pTrigonometricMoment[2]+float_complex(0.0f,1.25663706f)*pOutExponentialMoment[4]*pTrigonometricMoment[1];
	pOutExponentialMoment[6]=float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]*pTrigonometricMoment[6]+float_complex(0.0f,5.23598776f)*pOutExponentialMoment[1]*pTrigonometricMoment[5]+float_complex(0.0f,4.1887902f)*pOutExponentialMoment[2]*pTrigonometricMoment[4]+float_complex(0.0f,3.14159265f)*pOutExponentialMoment[3]*pTrigonometricMoment[3]+float_complex(0.0f,2.0943951f)*pOutExponentialMoment[4]*pTrigonometricMoment[2]+float_complex(0.0f,1.04719755f)*pOutExponentialMoment[5]*pTrigonometricMoment[1];
	pOutExponentialMoment[7]=float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]*pTrigonometricMoment[7]+float_complex(0.0f,5.38558741f)*pOutExponentialMoment[1]*pTrigonometricMoment[6]+float_complex(0.0f,4.48798951f)*pOutExponentialMoment[2]*pTrigonometricMoment[5]+float_complex(0.0f,3.5903916f)*pOutExponentialMoment[3]*pTrigonometricMoment[4]+float_complex(0.0f,2.6927937f)*pOutExponentialMoment[4]*pTrigonometricMoment[3]+float_complex(0.0f,1.7951958f)*pOutExponentialMoment[5]*pTrigonometricMoment[2]+float_complex(0.0f,0.897597901f)*pOutExponentialMoment[6]*pTrigonometricMoment[1];
	pOutExponentialMoment[8]=float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]*pTrigonometricMoment[8]+float_complex(0.0f,5.49778714f)*pOutExponentialMoment[1]*pTrigonometricMoment[7]+float_complex(0.0f,4.71238898f)*pOutExponentialMoment[2]*pTrigonometricMoment[6]+float_complex(0.0f,3.92699082f)*pOutExponentialMoment[3]*pTrigonometricMoment[5]+float_complex(0.0f,3.14159265f)*pOutExponentialMoment[4]*pTrigonometricMoment[4]+float_complex(0.0f,2.35619449f)*pOutExponentialMoment[5]*pTrigonometricMoment[3]+float_complex(0.0f,1.57079633f)*pOutExponentialMoment[6]*pTrigonometricMoment[2]+float_complex(0.0f,0.785398163f)*pOutExponentialMoment[7]*pTrigonometricMoment[1];
	pOutExponentialMoment[9]=float_complex(0.0f,6.28318531f)*pOutExponentialMoment[0]*pTrigonometricMoment[9]+float_complex(0.0f,5.58505361f)*pOutExponentialMoment[1]*pTrigonometricMoment[8]+float_complex(0.0f,4.88692191f)*pOutExponentialMoment[2]*pTrigonometricMoment[7]+float_complex(0.0f,4.1887902f)*pOutExponentialMoment[3]*pTrigonometricMoment[6]+float_complex(0.0f,3.4906585f)*pOutExponentialMoment[4]*pTrigonometricMoment[5]+float_complex(0.0f,2.7925268f)*pOutExponentialMoment[5]*pTrigonometricMoment[4]+float_complex(0.0f,2.0943951f)*pOutExponentialMoment[6]*pTrigonometricMoment[3]+float_complex(0.0f,1.3962634f)*pOutExponentialMoment[7]*pTrigonometricMoment[2]+float_complex(0.0f,0.698131701f)*pOutExponentialMoment[8]*pTrigonometricMoment[1];
	pOutExponentialMoment[0]=2.0f*pOutExponentialMoment[0];
}


FUNC void levinsonsAlgorithm4(float_complex pOutSolution[4],const float_complex pFirstColumn[4]){
	pOutSolution[0]=float_complex(1.0f/(pFirstColumn[0].x),0.0f);
	float_complex unnormalizedNextMomentCenter;
	float_complex dotProduct;
	float dotProductAbsSquare;
	float_complex flippedSolution1;
	float_complex flippedSolution2;
	float_complex flippedSolution3;
	float factor;
	unnormalizedNextMomentCenter=float_complex(0.0f,0.0f);
	dotProduct=pOutSolution[0].x*pFirstColumn[1]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	flippedSolution1=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(-flippedSolution1.x*dotProduct);
	unnormalizedNextMomentCenter=pOutSolution[1]*pFirstColumn[1];
	dotProduct=pOutSolution[0].x*pFirstColumn[2]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	flippedSolution1=conjugate(pOutSolution[1]);
	flippedSolution2=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(pOutSolution[1]-flippedSolution1*dotProduct);
	pOutSolution[2]=factor*(-flippedSolution2.x*dotProduct);
	unnormalizedNextMomentCenter=pOutSolution[1]*pFirstColumn[2]+pOutSolution[2]*pFirstColumn[1];
	dotProduct=pOutSolution[0].x*pFirstColumn[3]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	flippedSolution1=conjugate(pOutSolution[2]);
	flippedSolution2=conjugate(pOutSolution[1]);
	flippedSolution3=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(pOutSolution[1]-flippedSolution1*dotProduct);
	pOutSolution[2]=factor*(pOutSolution[2]-flippedSolution2*dotProduct);
	pOutSolution[3]=factor*(-flippedSolution3.x*dotProduct);
}

FUNC void levinsonsAlgorithmWithBiasing4(float_complex pOutSolution[4],float_complex pFirstColumn[4]){
	float oneMinusCorrectionBias=0.9999f;
	float correctedFactor=1.0f/(1.0f-oneMinusCorrectionBias*oneMinusCorrectionBias);
	pOutSolution[0]=float_complex(1.0f/(pFirstColumn[0].x),0.0f);
	float_complex unnormalizedNextMomentCenter;
	float_complex dotProduct;
	float dotProductAbsSquare;
	float_complex flippedSolution1;
	float_complex flippedSolution2;
	float_complex flippedSolution3;
	float factor;
	unnormalizedNextMomentCenter=float_complex(0.0f,0.0f);
	dotProduct=pOutSolution[0].x*pFirstColumn[1]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	if(factor<0.0f){
		dotProduct=(oneMinusCorrectionBias*1.0f/std::sqrt(dotProductAbsSquare))*dotProduct;
		pFirstColumn[1]=(dotProduct-unnormalizedNextMomentCenter)/(pOutSolution[0].x);
		factor=correctedFactor;
		oneMinusCorrectionBias=0.0f;
		correctedFactor=1.0f;
	}
	
	flippedSolution1=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(-flippedSolution1.x*dotProduct);
	unnormalizedNextMomentCenter=pOutSolution[1]*pFirstColumn[1];
	dotProduct=pOutSolution[0].x*pFirstColumn[2]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	if(factor<0.0f){
		dotProduct=(oneMinusCorrectionBias*1.0f/std::sqrt(dotProductAbsSquare))*dotProduct;
		pFirstColumn[2]=(dotProduct-unnormalizedNextMomentCenter)/(pOutSolution[0].x);
		factor=correctedFactor;
		oneMinusCorrectionBias=0.0f;
		correctedFactor=1.0f;
	}
	
	flippedSolution1=conjugate(pOutSolution[1]);
	flippedSolution2=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(pOutSolution[1]-flippedSolution1*dotProduct);
	pOutSolution[2]=factor*(-flippedSolution2.x*dotProduct);
	unnormalizedNextMomentCenter=pOutSolution[1]*pFirstColumn[2]+pOutSolution[2]*pFirstColumn[1];
	dotProduct=pOutSolution[0].x*pFirstColumn[3]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	if(factor<0.0f){
		dotProduct=(oneMinusCorrectionBias*1.0f/std::sqrt(dotProductAbsSquare))*dotProduct;
		pFirstColumn[3]=(dotProduct-unnormalizedNextMomentCenter)/(pOutSolution[0].x);
		factor=correctedFactor;
	}
	
	flippedSolution1=conjugate(pOutSolution[2]);
	flippedSolution2=conjugate(pOutSolution[1]);
	flippedSolution3=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(pOutSolution[1]-flippedSolution1*dotProduct);
	pOutSolution[2]=factor*(pOutSolution[2]-flippedSolution2*dotProduct);
	pOutSolution[3]=factor*(-flippedSolution3.x*dotProduct);
}

FUNC void levinsonsAlgorithm10(float_complex pOutSolution[10],const float_complex pFirstColumn[10]){
	pOutSolution[0]=float_complex(1.0f/(pFirstColumn[0].x),0.0f);
	float_complex unnormalizedNextMomentCenter;
	float_complex dotProduct;
	float dotProductAbsSquare;
	float_complex flippedSolution1;
	float_complex flippedSolution2;
	float_complex flippedSolution3;
	float_complex flippedSolution4;
	float_complex flippedSolution5;
	float_complex flippedSolution6;
	float_complex flippedSolution7;
	float_complex flippedSolution8;
	float_complex flippedSolution9;
	float factor;
	unnormalizedNextMomentCenter=float_complex(0.0f,0.0f);
	dotProduct=pOutSolution[0].x*pFirstColumn[1]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	flippedSolution1=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(-flippedSolution1.x*dotProduct);
	unnormalizedNextMomentCenter=pOutSolution[1]*pFirstColumn[1];
	dotProduct=pOutSolution[0].x*pFirstColumn[2]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	flippedSolution1=conjugate(pOutSolution[1]);
	flippedSolution2=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(pOutSolution[1]-flippedSolution1*dotProduct);
	pOutSolution[2]=factor*(-flippedSolution2.x*dotProduct);
	unnormalizedNextMomentCenter=pOutSolution[1]*pFirstColumn[2]+pOutSolution[2]*pFirstColumn[1];
	dotProduct=pOutSolution[0].x*pFirstColumn[3]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	flippedSolution1=conjugate(pOutSolution[2]);
	flippedSolution2=conjugate(pOutSolution[1]);
	flippedSolution3=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(pOutSolution[1]-flippedSolution1*dotProduct);
	pOutSolution[2]=factor*(pOutSolution[2]-flippedSolution2*dotProduct);
	pOutSolution[3]=factor*(-flippedSolution3.x*dotProduct);
	unnormalizedNextMomentCenter=pOutSolution[1]*pFirstColumn[3]+pOutSolution[2]*pFirstColumn[2]+pOutSolution[3]*pFirstColumn[1];
	dotProduct=pOutSolution[0].x*pFirstColumn[4]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	flippedSolution1=conjugate(pOutSolution[3]);
	flippedSolution2=conjugate(pOutSolution[2]);
	flippedSolution3=conjugate(pOutSolution[1]);
	flippedSolution4=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(pOutSolution[1]-flippedSolution1*dotProduct);
	pOutSolution[2]=factor*(pOutSolution[2]-flippedSolution2*dotProduct);
	pOutSolution[3]=factor*(pOutSolution[3]-flippedSolution3*dotProduct);
	pOutSolution[4]=factor*(-flippedSolution4.x*dotProduct);
	unnormalizedNextMomentCenter=pOutSolution[1]*pFirstColumn[4]+pOutSolution[2]*pFirstColumn[3]+pOutSolution[3]*pFirstColumn[2]+pOutSolution[4]*pFirstColumn[1];
	dotProduct=pOutSolution[0].x*pFirstColumn[5]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	flippedSolution1=conjugate(pOutSolution[4]);
	flippedSolution2=conjugate(pOutSolution[3]);
	flippedSolution3=conjugate(pOutSolution[2]);
	flippedSolution4=conjugate(pOutSolution[1]);
	flippedSolution5=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(pOutSolution[1]-flippedSolution1*dotProduct);
	pOutSolution[2]=factor*(pOutSolution[2]-flippedSolution2*dotProduct);
	pOutSolution[3]=factor*(pOutSolution[3]-flippedSolution3*dotProduct);
	pOutSolution[4]=factor*(pOutSolution[4]-flippedSolution4*dotProduct);
	pOutSolution[5]=factor*(-flippedSolution5.x*dotProduct);
	unnormalizedNextMomentCenter=pOutSolution[1]*pFirstColumn[5]+pOutSolution[2]*pFirstColumn[4]+pOutSolution[3]*pFirstColumn[3]+pOutSolution[4]*pFirstColumn[2]+pOutSolution[5]*pFirstColumn[1];
	dotProduct=pOutSolution[0].x*pFirstColumn[6]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	flippedSolution1=conjugate(pOutSolution[5]);
	flippedSolution2=conjugate(pOutSolution[4]);
	flippedSolution3=conjugate(pOutSolution[3]);
	flippedSolution4=conjugate(pOutSolution[2]);
	flippedSolution5=conjugate(pOutSolution[1]);
	flippedSolution6=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(pOutSolution[1]-flippedSolution1*dotProduct);
	pOutSolution[2]=factor*(pOutSolution[2]-flippedSolution2*dotProduct);
	pOutSolution[3]=factor*(pOutSolution[3]-flippedSolution3*dotProduct);
	pOutSolution[4]=factor*(pOutSolution[4]-flippedSolution4*dotProduct);
	pOutSolution[5]=factor*(pOutSolution[5]-flippedSolution5*dotProduct);
	pOutSolution[6]=factor*(-flippedSolution6.x*dotProduct);
	unnormalizedNextMomentCenter=pOutSolution[1]*pFirstColumn[6]+pOutSolution[2]*pFirstColumn[5]+pOutSolution[3]*pFirstColumn[4]+pOutSolution[4]*pFirstColumn[3]+pOutSolution[5]*pFirstColumn[2]+pOutSolution[6]*pFirstColumn[1];
	dotProduct=pOutSolution[0].x*pFirstColumn[7]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	flippedSolution1=conjugate(pOutSolution[6]);
	flippedSolution2=conjugate(pOutSolution[5]);
	flippedSolution3=conjugate(pOutSolution[4]);
	flippedSolution4=conjugate(pOutSolution[3]);
	flippedSolution5=conjugate(pOutSolution[2]);
	flippedSolution6=conjugate(pOutSolution[1]);
	flippedSolution7=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(pOutSolution[1]-flippedSolution1*dotProduct);
	pOutSolution[2]=factor*(pOutSolution[2]-flippedSolution2*dotProduct);
	pOutSolution[3]=factor*(pOutSolution[3]-flippedSolution3*dotProduct);
	pOutSolution[4]=factor*(pOutSolution[4]-flippedSolution4*dotProduct);
	pOutSolution[5]=factor*(pOutSolution[5]-flippedSolution5*dotProduct);
	pOutSolution[6]=factor*(pOutSolution[6]-flippedSolution6*dotProduct);
	pOutSolution[7]=factor*(-flippedSolution7.x*dotProduct);
	unnormalizedNextMomentCenter=pOutSolution[1]*pFirstColumn[7]+pOutSolution[2]*pFirstColumn[6]+pOutSolution[3]*pFirstColumn[5]+pOutSolution[4]*pFirstColumn[4]+pOutSolution[5]*pFirstColumn[3]+pOutSolution[6]*pFirstColumn[2]+pOutSolution[7]*pFirstColumn[1];
	dotProduct=pOutSolution[0].x*pFirstColumn[8]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	flippedSolution1=conjugate(pOutSolution[7]);
	flippedSolution2=conjugate(pOutSolution[6]);
	flippedSolution3=conjugate(pOutSolution[5]);
	flippedSolution4=conjugate(pOutSolution[4]);
	flippedSolution5=conjugate(pOutSolution[3]);
	flippedSolution6=conjugate(pOutSolution[2]);
	flippedSolution7=conjugate(pOutSolution[1]);
	flippedSolution8=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(pOutSolution[1]-flippedSolution1*dotProduct);
	pOutSolution[2]=factor*(pOutSolution[2]-flippedSolution2*dotProduct);
	pOutSolution[3]=factor*(pOutSolution[3]-flippedSolution3*dotProduct);
	pOutSolution[4]=factor*(pOutSolution[4]-flippedSolution4*dotProduct);
	pOutSolution[5]=factor*(pOutSolution[5]-flippedSolution5*dotProduct);
	pOutSolution[6]=factor*(pOutSolution[6]-flippedSolution6*dotProduct);
	pOutSolution[7]=factor*(pOutSolution[7]-flippedSolution7*dotProduct);
	pOutSolution[8]=factor*(-flippedSolution8.x*dotProduct);
	unnormalizedNextMomentCenter=pOutSolution[1]*pFirstColumn[8]+pOutSolution[2]*pFirstColumn[7]+pOutSolution[3]*pFirstColumn[6]+pOutSolution[4]*pFirstColumn[5]+pOutSolution[5]*pFirstColumn[4]+pOutSolution[6]*pFirstColumn[3]+pOutSolution[7]*pFirstColumn[2]+pOutSolution[8]*pFirstColumn[1];
	dotProduct=pOutSolution[0].x*pFirstColumn[9]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	flippedSolution1=conjugate(pOutSolution[8]);
	flippedSolution2=conjugate(pOutSolution[7]);
	flippedSolution3=conjugate(pOutSolution[6]);
	flippedSolution4=conjugate(pOutSolution[5]);
	flippedSolution5=conjugate(pOutSolution[4]);
	flippedSolution6=conjugate(pOutSolution[3]);
	flippedSolution7=conjugate(pOutSolution[2]);
	flippedSolution8=conjugate(pOutSolution[1]);
	flippedSolution9=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(pOutSolution[1]-flippedSolution1*dotProduct);
	pOutSolution[2]=factor*(pOutSolution[2]-flippedSolution2*dotProduct);
	pOutSolution[3]=factor*(pOutSolution[3]-flippedSolution3*dotProduct);
	pOutSolution[4]=factor*(pOutSolution[4]-flippedSolution4*dotProduct);
	pOutSolution[5]=factor*(pOutSolution[5]-flippedSolution5*dotProduct);
	pOutSolution[6]=factor*(pOutSolution[6]-flippedSolution6*dotProduct);
	pOutSolution[7]=factor*(pOutSolution[7]-flippedSolution7*dotProduct);
	pOutSolution[8]=factor*(pOutSolution[8]-flippedSolution8*dotProduct);
	pOutSolution[9]=factor*(-flippedSolution9.x*dotProduct);
}

FUNC void levinsonsAlgorithmWithBiasing10(float_complex pOutSolution[10],float_complex pFirstColumn[10]){
	float oneMinusCorrectionBias=0.9999f;
	float correctedFactor=1.0f/(1.0f-oneMinusCorrectionBias*oneMinusCorrectionBias);
	pOutSolution[0]=float_complex(1.0f/(pFirstColumn[0].x),0.0f);
	float_complex unnormalizedNextMomentCenter;
	float_complex dotProduct;
	float dotProductAbsSquare;
	float_complex flippedSolution1;
	float_complex flippedSolution2;
	float_complex flippedSolution3;
	float_complex flippedSolution4;
	float_complex flippedSolution5;
	float_complex flippedSolution6;
	float_complex flippedSolution7;
	float_complex flippedSolution8;
	float_complex flippedSolution9;
	float factor;
	unnormalizedNextMomentCenter=float_complex(0.0f,0.0f);
	dotProduct=pOutSolution[0].x*pFirstColumn[1]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	if(factor<0.0f){
		dotProduct=(oneMinusCorrectionBias*1.0f/std::sqrt(dotProductAbsSquare))*dotProduct;
		pFirstColumn[1]=(dotProduct-unnormalizedNextMomentCenter)/(pOutSolution[0].x);
		factor=correctedFactor;
		oneMinusCorrectionBias=0.0f;
		correctedFactor=1.0f;
	}
	
	flippedSolution1=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(-flippedSolution1.x*dotProduct);
	unnormalizedNextMomentCenter=pOutSolution[1]*pFirstColumn[1];
	dotProduct=pOutSolution[0].x*pFirstColumn[2]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	if(factor<0.0f){
		dotProduct=(oneMinusCorrectionBias*1.0f/std::sqrt(dotProductAbsSquare))*dotProduct;
		pFirstColumn[2]=(dotProduct-unnormalizedNextMomentCenter)/(pOutSolution[0].x);
		factor=correctedFactor;
		oneMinusCorrectionBias=0.0f;
		correctedFactor=1.0f;
	}
	
	flippedSolution1=conjugate(pOutSolution[1]);
	flippedSolution2=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(pOutSolution[1]-flippedSolution1*dotProduct);
	pOutSolution[2]=factor*(-flippedSolution2.x*dotProduct);
	unnormalizedNextMomentCenter=pOutSolution[1]*pFirstColumn[2]+pOutSolution[2]*pFirstColumn[1];
	dotProduct=pOutSolution[0].x*pFirstColumn[3]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	if(factor<0.0f){
		dotProduct=(oneMinusCorrectionBias*1.0f/std::sqrt(dotProductAbsSquare))*dotProduct;
		pFirstColumn[3]=(dotProduct-unnormalizedNextMomentCenter)/(pOutSolution[0].x);
		factor=correctedFactor;
		oneMinusCorrectionBias=0.0f;
		correctedFactor=1.0f;
	}
	
	flippedSolution1=conjugate(pOutSolution[2]);
	flippedSolution2=conjugate(pOutSolution[1]);
	flippedSolution3=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(pOutSolution[1]-flippedSolution1*dotProduct);
	pOutSolution[2]=factor*(pOutSolution[2]-flippedSolution2*dotProduct);
	pOutSolution[3]=factor*(-flippedSolution3.x*dotProduct);
	unnormalizedNextMomentCenter=pOutSolution[1]*pFirstColumn[3]+pOutSolution[2]*pFirstColumn[2]+pOutSolution[3]*pFirstColumn[1];
	dotProduct=pOutSolution[0].x*pFirstColumn[4]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	if(factor<0.0f){
		dotProduct=(oneMinusCorrectionBias*1.0f/std::sqrt(dotProductAbsSquare))*dotProduct;
		pFirstColumn[4]=(dotProduct-unnormalizedNextMomentCenter)/(pOutSolution[0].x);
		factor=correctedFactor;
		oneMinusCorrectionBias=0.0f;
		correctedFactor=1.0f;
	}
	
	flippedSolution1=conjugate(pOutSolution[3]);
	flippedSolution2=conjugate(pOutSolution[2]);
	flippedSolution3=conjugate(pOutSolution[1]);
	flippedSolution4=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(pOutSolution[1]-flippedSolution1*dotProduct);
	pOutSolution[2]=factor*(pOutSolution[2]-flippedSolution2*dotProduct);
	pOutSolution[3]=factor*(pOutSolution[3]-flippedSolution3*dotProduct);
	pOutSolution[4]=factor*(-flippedSolution4.x*dotProduct);
	unnormalizedNextMomentCenter=pOutSolution[1]*pFirstColumn[4]+pOutSolution[2]*pFirstColumn[3]+pOutSolution[3]*pFirstColumn[2]+pOutSolution[4]*pFirstColumn[1];
	dotProduct=pOutSolution[0].x*pFirstColumn[5]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	if(factor<0.0f){
		dotProduct=(oneMinusCorrectionBias*1.0f/std::sqrt(dotProductAbsSquare))*dotProduct;
		pFirstColumn[5]=(dotProduct-unnormalizedNextMomentCenter)/(pOutSolution[0].x);
		factor=correctedFactor;
		oneMinusCorrectionBias=0.0f;
		correctedFactor=1.0f;
	}
	
	flippedSolution1=conjugate(pOutSolution[4]);
	flippedSolution2=conjugate(pOutSolution[3]);
	flippedSolution3=conjugate(pOutSolution[2]);
	flippedSolution4=conjugate(pOutSolution[1]);
	flippedSolution5=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(pOutSolution[1]-flippedSolution1*dotProduct);
	pOutSolution[2]=factor*(pOutSolution[2]-flippedSolution2*dotProduct);
	pOutSolution[3]=factor*(pOutSolution[3]-flippedSolution3*dotProduct);
	pOutSolution[4]=factor*(pOutSolution[4]-flippedSolution4*dotProduct);
	pOutSolution[5]=factor*(-flippedSolution5.x*dotProduct);
	unnormalizedNextMomentCenter=pOutSolution[1]*pFirstColumn[5]+pOutSolution[2]*pFirstColumn[4]+pOutSolution[3]*pFirstColumn[3]+pOutSolution[4]*pFirstColumn[2]+pOutSolution[5]*pFirstColumn[1];
	dotProduct=pOutSolution[0].x*pFirstColumn[6]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	if(factor<0.0f){
		dotProduct=(oneMinusCorrectionBias*1.0f/std::sqrt(dotProductAbsSquare))*dotProduct;
		pFirstColumn[6]=(dotProduct-unnormalizedNextMomentCenter)/(pOutSolution[0].x);
		factor=correctedFactor;
		oneMinusCorrectionBias=0.0f;
		correctedFactor=1.0f;
	}
	
	flippedSolution1=conjugate(pOutSolution[5]);
	flippedSolution2=conjugate(pOutSolution[4]);
	flippedSolution3=conjugate(pOutSolution[3]);
	flippedSolution4=conjugate(pOutSolution[2]);
	flippedSolution5=conjugate(pOutSolution[1]);
	flippedSolution6=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(pOutSolution[1]-flippedSolution1*dotProduct);
	pOutSolution[2]=factor*(pOutSolution[2]-flippedSolution2*dotProduct);
	pOutSolution[3]=factor*(pOutSolution[3]-flippedSolution3*dotProduct);
	pOutSolution[4]=factor*(pOutSolution[4]-flippedSolution4*dotProduct);
	pOutSolution[5]=factor*(pOutSolution[5]-flippedSolution5*dotProduct);
	pOutSolution[6]=factor*(-flippedSolution6.x*dotProduct);
	unnormalizedNextMomentCenter=pOutSolution[1]*pFirstColumn[6]+pOutSolution[2]*pFirstColumn[5]+pOutSolution[3]*pFirstColumn[4]+pOutSolution[4]*pFirstColumn[3]+pOutSolution[5]*pFirstColumn[2]+pOutSolution[6]*pFirstColumn[1];
	dotProduct=pOutSolution[0].x*pFirstColumn[7]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	if(factor<0.0f){
		dotProduct=(oneMinusCorrectionBias*1.0f/std::sqrt(dotProductAbsSquare))*dotProduct;
		pFirstColumn[7]=(dotProduct-unnormalizedNextMomentCenter)/(pOutSolution[0].x);
		factor=correctedFactor;
		oneMinusCorrectionBias=0.0f;
		correctedFactor=1.0f;
	}
	
	flippedSolution1=conjugate(pOutSolution[6]);
	flippedSolution2=conjugate(pOutSolution[5]);
	flippedSolution3=conjugate(pOutSolution[4]);
	flippedSolution4=conjugate(pOutSolution[3]);
	flippedSolution5=conjugate(pOutSolution[2]);
	flippedSolution6=conjugate(pOutSolution[1]);
	flippedSolution7=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(pOutSolution[1]-flippedSolution1*dotProduct);
	pOutSolution[2]=factor*(pOutSolution[2]-flippedSolution2*dotProduct);
	pOutSolution[3]=factor*(pOutSolution[3]-flippedSolution3*dotProduct);
	pOutSolution[4]=factor*(pOutSolution[4]-flippedSolution4*dotProduct);
	pOutSolution[5]=factor*(pOutSolution[5]-flippedSolution5*dotProduct);
	pOutSolution[6]=factor*(pOutSolution[6]-flippedSolution6*dotProduct);
	pOutSolution[7]=factor*(-flippedSolution7.x*dotProduct);
	unnormalizedNextMomentCenter=pOutSolution[1]*pFirstColumn[7]+pOutSolution[2]*pFirstColumn[6]+pOutSolution[3]*pFirstColumn[5]+pOutSolution[4]*pFirstColumn[4]+pOutSolution[5]*pFirstColumn[3]+pOutSolution[6]*pFirstColumn[2]+pOutSolution[7]*pFirstColumn[1];
	dotProduct=pOutSolution[0].x*pFirstColumn[8]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	if(factor<0.0f){
		dotProduct=(oneMinusCorrectionBias*1.0f/std::sqrt(dotProductAbsSquare))*dotProduct;
		pFirstColumn[8]=(dotProduct-unnormalizedNextMomentCenter)/(pOutSolution[0].x);
		factor=correctedFactor;
		oneMinusCorrectionBias=0.0f;
		correctedFactor=1.0f;
	}
	
	flippedSolution1=conjugate(pOutSolution[7]);
	flippedSolution2=conjugate(pOutSolution[6]);
	flippedSolution3=conjugate(pOutSolution[5]);
	flippedSolution4=conjugate(pOutSolution[4]);
	flippedSolution5=conjugate(pOutSolution[3]);
	flippedSolution6=conjugate(pOutSolution[2]);
	flippedSolution7=conjugate(pOutSolution[1]);
	flippedSolution8=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(pOutSolution[1]-flippedSolution1*dotProduct);
	pOutSolution[2]=factor*(pOutSolution[2]-flippedSolution2*dotProduct);
	pOutSolution[3]=factor*(pOutSolution[3]-flippedSolution3*dotProduct);
	pOutSolution[4]=factor*(pOutSolution[4]-flippedSolution4*dotProduct);
	pOutSolution[5]=factor*(pOutSolution[5]-flippedSolution5*dotProduct);
	pOutSolution[6]=factor*(pOutSolution[6]-flippedSolution6*dotProduct);
	pOutSolution[7]=factor*(pOutSolution[7]-flippedSolution7*dotProduct);
	pOutSolution[8]=factor*(-flippedSolution8.x*dotProduct);
	unnormalizedNextMomentCenter=pOutSolution[1]*pFirstColumn[8]+pOutSolution[2]*pFirstColumn[7]+pOutSolution[3]*pFirstColumn[6]+pOutSolution[4]*pFirstColumn[5]+pOutSolution[5]*pFirstColumn[4]+pOutSolution[6]*pFirstColumn[3]+pOutSolution[7]*pFirstColumn[2]+pOutSolution[8]*pFirstColumn[1];
	dotProduct=pOutSolution[0].x*pFirstColumn[9]+unnormalizedNextMomentCenter;
	dotProductAbsSquare=absSqr(dotProduct);
	factor=1.0f/(1.0f-dotProductAbsSquare);
	if(factor<0.0f){
		dotProduct=(oneMinusCorrectionBias*1.0f/std::sqrt(dotProductAbsSquare))*dotProduct;
		pFirstColumn[9]=(dotProduct-unnormalizedNextMomentCenter)/(pOutSolution[0].x);
		factor=correctedFactor;
	}
	
	flippedSolution1=conjugate(pOutSolution[8]);
	flippedSolution2=conjugate(pOutSolution[7]);
	flippedSolution3=conjugate(pOutSolution[6]);
	flippedSolution4=conjugate(pOutSolution[5]);
	flippedSolution5=conjugate(pOutSolution[4]);
	flippedSolution6=conjugate(pOutSolution[3]);
	flippedSolution7=conjugate(pOutSolution[2]);
	flippedSolution8=conjugate(pOutSolution[1]);
	flippedSolution9=float_complex(pOutSolution[0].x,0.0f);
	pOutSolution[0]=float_complex(factor*pOutSolution[0].x,0.0f);
	pOutSolution[1]=factor*(pOutSolution[1]-flippedSolution1*dotProduct);
	pOutSolution[2]=factor*(pOutSolution[2]-flippedSolution2*dotProduct);
	pOutSolution[3]=factor*(pOutSolution[3]-flippedSolution3*dotProduct);
	pOutSolution[4]=factor*(pOutSolution[4]-flippedSolution4*dotProduct);
	pOutSolution[5]=factor*(pOutSolution[5]-flippedSolution5*dotProduct);
	pOutSolution[6]=factor*(pOutSolution[6]-flippedSolution6*dotProduct);
	pOutSolution[7]=factor*(pOutSolution[7]-flippedSolution7*dotProduct);
	pOutSolution[8]=factor*(pOutSolution[8]-flippedSolution8*dotProduct);
	pOutSolution[9]=factor*(-flippedSolution9.x*dotProduct);
}


FUNC void computeAutocorrelation4(float_complex pOutAutocorrelation[4],const float_complex pSignal[4]){
	pOutAutocorrelation[0]=pSignal[0]*conjugate(pSignal[0])+pSignal[1]*conjugate(pSignal[1])+pSignal[2]*conjugate(pSignal[2])+pSignal[3]*conjugate(pSignal[3]);
	pOutAutocorrelation[1]=pSignal[0]*conjugate(pSignal[1])+pSignal[1]*conjugate(pSignal[2])+pSignal[2]*conjugate(pSignal[3]);
	pOutAutocorrelation[2]=pSignal[0]*conjugate(pSignal[2])+pSignal[1]*conjugate(pSignal[3]);
	pOutAutocorrelation[3]=pSignal[0]*conjugate(pSignal[3]);
}

FUNC void computeAutocorrelation10(float_complex pOutAutocorrelation[10],const float_complex pSignal[10]){
	pOutAutocorrelation[0]=pSignal[0]*conjugate(pSignal[0])+pSignal[1]*conjugate(pSignal[1])+pSignal[2]*conjugate(pSignal[2])+pSignal[3]*conjugate(pSignal[3])+pSignal[4]*conjugate(pSignal[4])+pSignal[5]*conjugate(pSignal[5])+pSignal[6]*conjugate(pSignal[6])+pSignal[7]*conjugate(pSignal[7])+pSignal[8]*conjugate(pSignal[8])+pSignal[9]*conjugate(pSignal[9]);
	pOutAutocorrelation[1]=pSignal[0]*conjugate(pSignal[1])+pSignal[1]*conjugate(pSignal[2])+pSignal[2]*conjugate(pSignal[3])+pSignal[3]*conjugate(pSignal[4])+pSignal[4]*conjugate(pSignal[5])+pSignal[5]*conjugate(pSignal[6])+pSignal[6]*conjugate(pSignal[7])+pSignal[7]*conjugate(pSignal[8])+pSignal[8]*conjugate(pSignal[9]);
	pOutAutocorrelation[2]=pSignal[0]*conjugate(pSignal[2])+pSignal[1]*conjugate(pSignal[3])+pSignal[2]*conjugate(pSignal[4])+pSignal[3]*conjugate(pSignal[5])+pSignal[4]*conjugate(pSignal[6])+pSignal[5]*conjugate(pSignal[7])+pSignal[6]*conjugate(pSignal[8])+pSignal[7]*conjugate(pSignal[9]);
	pOutAutocorrelation[3]=pSignal[0]*conjugate(pSignal[3])+pSignal[1]*conjugate(pSignal[4])+pSignal[2]*conjugate(pSignal[5])+pSignal[3]*conjugate(pSignal[6])+pSignal[4]*conjugate(pSignal[7])+pSignal[5]*conjugate(pSignal[8])+pSignal[6]*conjugate(pSignal[9]);
	pOutAutocorrelation[4]=pSignal[0]*conjugate(pSignal[4])+pSignal[1]*conjugate(pSignal[5])+pSignal[2]*conjugate(pSignal[6])+pSignal[3]*conjugate(pSignal[7])+pSignal[4]*conjugate(pSignal[8])+pSignal[5]*conjugate(pSignal[9]);
	pOutAutocorrelation[5]=pSignal[0]*conjugate(pSignal[5])+pSignal[1]*conjugate(pSignal[6])+pSignal[2]*conjugate(pSignal[7])+pSignal[3]*conjugate(pSignal[8])+pSignal[4]*conjugate(pSignal[9]);
	pOutAutocorrelation[6]=pSignal[0]*conjugate(pSignal[6])+pSignal[1]*conjugate(pSignal[7])+pSignal[2]*conjugate(pSignal[8])+pSignal[3]*conjugate(pSignal[9]);
	pOutAutocorrelation[7]=pSignal[0]*conjugate(pSignal[7])+pSignal[1]*conjugate(pSignal[8])+pSignal[2]*conjugate(pSignal[9]);
	pOutAutocorrelation[8]=pSignal[0]*conjugate(pSignal[8])+pSignal[1]*conjugate(pSignal[9]);
	pOutAutocorrelation[9]=pSignal[0]*conjugate(pSignal[9]);
}


FUNC void computeCorrelation4(float_complex pOutCorrelation[4],const float_complex pLHS[4],const float_complex pRHS[4]){
	pOutCorrelation[0]=pLHS[0]*pRHS[0]+pLHS[1]*pRHS[1]+pLHS[2]*pRHS[2]+pLHS[3]*pRHS[3];
	pOutCorrelation[1]=pLHS[1]*pRHS[0]+pLHS[2]*pRHS[1]+pLHS[3]*pRHS[2];
	pOutCorrelation[2]=pLHS[2]*pRHS[0]+pLHS[3]*pRHS[1];
	pOutCorrelation[3]=pLHS[3]*pRHS[0];
}

FUNC void computeCorrelation10(float_complex pOutCorrelation[10],const float_complex pLHS[10],const float_complex pRHS[10]){
	pOutCorrelation[0]=pLHS[0]*pRHS[0]+pLHS[1]*pRHS[1]+pLHS[2]*pRHS[2]+pLHS[3]*pRHS[3]+pLHS[4]*pRHS[4]+pLHS[5]*pRHS[5]+pLHS[6]*pRHS[6]+pLHS[7]*pRHS[7]+pLHS[8]*pRHS[8]+pLHS[9]*pRHS[9];
	pOutCorrelation[1]=pLHS[1]*pRHS[0]+pLHS[2]*pRHS[1]+pLHS[3]*pRHS[2]+pLHS[4]*pRHS[3]+pLHS[5]*pRHS[4]+pLHS[6]*pRHS[5]+pLHS[7]*pRHS[6]+pLHS[8]*pRHS[7]+pLHS[9]*pRHS[8];
	pOutCorrelation[2]=pLHS[2]*pRHS[0]+pLHS[3]*pRHS[1]+pLHS[4]*pRHS[2]+pLHS[5]*pRHS[3]+pLHS[6]*pRHS[4]+pLHS[7]*pRHS[5]+pLHS[8]*pRHS[6]+pLHS[9]*pRHS[7];
	pOutCorrelation[3]=pLHS[3]*pRHS[0]+pLHS[4]*pRHS[1]+pLHS[5]*pRHS[2]+pLHS[6]*pRHS[3]+pLHS[7]*pRHS[4]+pLHS[8]*pRHS[5]+pLHS[9]*pRHS[6];
	pOutCorrelation[4]=pLHS[4]*pRHS[0]+pLHS[5]*pRHS[1]+pLHS[6]*pRHS[2]+pLHS[7]*pRHS[3]+pLHS[8]*pRHS[4]+pLHS[9]*pRHS[5];
	pOutCorrelation[5]=pLHS[5]*pRHS[0]+pLHS[6]*pRHS[1]+pLHS[7]*pRHS[2]+pLHS[8]*pRHS[3]+pLHS[9]*pRHS[4];
	pOutCorrelation[6]=pLHS[6]*pRHS[0]+pLHS[7]*pRHS[1]+pLHS[8]*pRHS[2]+pLHS[9]*pRHS[3];
	pOutCorrelation[7]=pLHS[7]*pRHS[0]+pLHS[8]*pRHS[1]+pLHS[9]*pRHS[2];
	pOutCorrelation[8]=pLHS[8]*pRHS[0]+pLHS[9]*pRHS[1];
	pOutCorrelation[9]=pLHS[9]*pRHS[0];
}


FUNC void computeImaginaryCorrelation4(float pOutCorrelation[4],const float_complex pLHS[4],const float_complex pRHS[4]){
	pOutCorrelation[0]=pLHS[0].x*pRHS[0].y+pLHS[0].y*pRHS[0].x+pLHS[1].x*pRHS[1].y+pLHS[1].y*pRHS[1].x+pLHS[2].x*pRHS[2].y+pLHS[2].y*pRHS[2].x+pLHS[3].x*pRHS[3].y+pLHS[3].y*pRHS[3].x;
	pOutCorrelation[1]=pLHS[1].x*pRHS[0].y+pLHS[1].y*pRHS[0].x+pLHS[2].x*pRHS[1].y+pLHS[2].y*pRHS[1].x+pLHS[3].x*pRHS[2].y+pLHS[3].y*pRHS[2].x;
	pOutCorrelation[2]=pLHS[2].x*pRHS[0].y+pLHS[2].y*pRHS[0].x+pLHS[3].x*pRHS[1].y+pLHS[3].y*pRHS[1].x;
	pOutCorrelation[3]=pLHS[3].x*pRHS[0].y+pLHS[3].y*pRHS[0].x;
}

FUNC void computeImaginaryCorrelation10(float pOutCorrelation[10],const float_complex pLHS[10],const float_complex pRHS[10]){
	pOutCorrelation[0]=pLHS[0].x*pRHS[0].y+pLHS[0].y*pRHS[0].x+pLHS[1].x*pRHS[1].y+pLHS[1].y*pRHS[1].x+pLHS[2].x*pRHS[2].y+pLHS[2].y*pRHS[2].x+pLHS[3].x*pRHS[3].y+pLHS[3].y*pRHS[3].x+pLHS[4].x*pRHS[4].y+pLHS[4].y*pRHS[4].x+pLHS[5].x*pRHS[5].y+pLHS[5].y*pRHS[5].x+pLHS[6].x*pRHS[6].y+pLHS[6].y*pRHS[6].x+pLHS[7].x*pRHS[7].y+pLHS[7].y*pRHS[7].x+pLHS[8].x*pRHS[8].y+pLHS[8].y*pRHS[8].x+pLHS[9].x*pRHS[9].y+pLHS[9].y*pRHS[9].x;
	pOutCorrelation[1]=pLHS[1].x*pRHS[0].y+pLHS[1].y*pRHS[0].x+pLHS[2].x*pRHS[1].y+pLHS[2].y*pRHS[1].x+pLHS[3].x*pRHS[2].y+pLHS[3].y*pRHS[2].x+pLHS[4].x*pRHS[3].y+pLHS[4].y*pRHS[3].x+pLHS[5].x*pRHS[4].y+pLHS[5].y*pRHS[4].x+pLHS[6].x*pRHS[5].y+pLHS[6].y*pRHS[5].x+pLHS[7].x*pRHS[6].y+pLHS[7].y*pRHS[6].x+pLHS[8].x*pRHS[7].y+pLHS[8].y*pRHS[7].x+pLHS[9].x*pRHS[8].y+pLHS[9].y*pRHS[8].x;
	pOutCorrelation[2]=pLHS[2].x*pRHS[0].y+pLHS[2].y*pRHS[0].x+pLHS[3].x*pRHS[1].y+pLHS[3].y*pRHS[1].x+pLHS[4].x*pRHS[2].y+pLHS[4].y*pRHS[2].x+pLHS[5].x*pRHS[3].y+pLHS[5].y*pRHS[3].x+pLHS[6].x*pRHS[4].y+pLHS[6].y*pRHS[4].x+pLHS[7].x*pRHS[5].y+pLHS[7].y*pRHS[5].x+pLHS[8].x*pRHS[6].y+pLHS[8].y*pRHS[6].x+pLHS[9].x*pRHS[7].y+pLHS[9].y*pRHS[7].x;
	pOutCorrelation[3]=pLHS[3].x*pRHS[0].y+pLHS[3].y*pRHS[0].x+pLHS[4].x*pRHS[1].y+pLHS[4].y*pRHS[1].x+pLHS[5].x*pRHS[2].y+pLHS[5].y*pRHS[2].x+pLHS[6].x*pRHS[3].y+pLHS[6].y*pRHS[3].x+pLHS[7].x*pRHS[4].y+pLHS[7].y*pRHS[4].x+pLHS[8].x*pRHS[5].y+pLHS[8].y*pRHS[5].x+pLHS[9].x*pRHS[6].y+pLHS[9].y*pRHS[6].x;
	pOutCorrelation[4]=pLHS[4].x*pRHS[0].y+pLHS[4].y*pRHS[0].x+pLHS[5].x*pRHS[1].y+pLHS[5].y*pRHS[1].x+pLHS[6].x*pRHS[2].y+pLHS[6].y*pRHS[2].x+pLHS[7].x*pRHS[3].y+pLHS[7].y*pRHS[3].x+pLHS[8].x*pRHS[4].y+pLHS[8].y*pRHS[4].x+pLHS[9].x*pRHS[5].y+pLHS[9].y*pRHS[5].x;
	pOutCorrelation[5]=pLHS[5].x*pRHS[0].y+pLHS[5].y*pRHS[0].x+pLHS[6].x*pRHS[1].y+pLHS[6].y*pRHS[1].x+pLHS[7].x*pRHS[2].y+pLHS[7].y*pRHS[2].x+pLHS[8].x*pRHS[3].y+pLHS[8].y*pRHS[3].x+pLHS[9].x*pRHS[4].y+pLHS[9].y*pRHS[4].x;
	pOutCorrelation[6]=pLHS[6].x*pRHS[0].y+pLHS[6].y*pRHS[0].x+pLHS[7].x*pRHS[1].y+pLHS[7].y*pRHS[1].x+pLHS[8].x*pRHS[2].y+pLHS[8].y*pRHS[2].x+pLHS[9].x*pRHS[3].y+pLHS[9].y*pRHS[3].x;
	pOutCorrelation[7]=pLHS[7].x*pRHS[0].y+pLHS[7].y*pRHS[0].x+pLHS[8].x*pRHS[1].y+pLHS[8].y*pRHS[1].x+pLHS[9].x*pRHS[2].y+pLHS[9].y*pRHS[2].x;
	pOutCorrelation[8]=pLHS[8].x*pRHS[0].y+pLHS[8].y*pRHS[0].x+pLHS[9].x*pRHS[1].y+pLHS[9].y*pRHS[1].x;
	pOutCorrelation[9]=pLHS[9].x*pRHS[0].y+pLHS[9].y*pRHS[0].x;
}


FUNC float_complex evaluateFastHerglotzTransform4(const float_complex circlePoint,const float_complex pExponentialMoment[4],const float_complex pEvaluationPolynomial[4]){
	float_complex conjCirclePoint=conjugate(circlePoint);
	float polynomial3=pEvaluationPolynomial[0].x;
	float_complex polynomial2=pEvaluationPolynomial[1]+polynomial3*conjCirclePoint;
	float_complex polynomial1=pEvaluationPolynomial[2]+conjCirclePoint*polynomial2;
	float_complex polynomial0=pEvaluationPolynomial[3]+conjCirclePoint*polynomial1;
	float_complex dotProduct=polynomial1*pExponentialMoment[1]+polynomial2*pExponentialMoment[2]+polynomial3*pExponentialMoment[3];
	return pExponentialMoment[0]+2.0f*(dotProduct)/(polynomial0);
}

FUNC float_complex evaluateFastHerglotzTransform10(const float_complex circlePoint,const float_complex pExponentialMoment[10],const float_complex pEvaluationPolynomial[10]){
	float_complex conjCirclePoint=conjugate(circlePoint);
	float polynomial9=pEvaluationPolynomial[0].x;
	float_complex polynomial8=pEvaluationPolynomial[1]+polynomial9*conjCirclePoint;
	float_complex polynomial7=pEvaluationPolynomial[2]+conjCirclePoint*polynomial8;
	float_complex polynomial6=pEvaluationPolynomial[3]+conjCirclePoint*polynomial7;
	float_complex polynomial5=pEvaluationPolynomial[4]+conjCirclePoint*polynomial6;
	float_complex polynomial4=pEvaluationPolynomial[5]+conjCirclePoint*polynomial5;
	float_complex polynomial3=pEvaluationPolynomial[6]+conjCirclePoint*polynomial4;
	float_complex polynomial2=pEvaluationPolynomial[7]+conjCirclePoint*polynomial3;
	float_complex polynomial1=pEvaluationPolynomial[8]+conjCirclePoint*polynomial2;
	float_complex polynomial0=pEvaluationPolynomial[9]+conjCirclePoint*polynomial1;
	float_complex dotProduct=polynomial1*pExponentialMoment[1]+polynomial2*pExponentialMoment[2]+polynomial3*pExponentialMoment[3]+polynomial4*pExponentialMoment[4]+polynomial5*pExponentialMoment[5]+polynomial6*pExponentialMoment[6]+polynomial7*pExponentialMoment[7]+polynomial8*pExponentialMoment[8]+polynomial9*pExponentialMoment[9];
	return pExponentialMoment[0]+2.0f*(dotProduct)/(polynomial0);
}


FUNC float evaluateFourierSeries4(const float_complex circlePoint,const float_complex pFourierCoefficient[4]){
	// Minimize sequential dependencies by computing powers of circlePoint
	// by multiplying previous powers by powers of two
	float_complex circlePointPower1=circlePoint;
	float_complex circlePointPower2=circlePointPower1*circlePointPower1;
	float_complex circlePointPower3=circlePointPower1*circlePointPower2;
	float result=pFourierCoefficient[1].x*circlePointPower1.x-pFourierCoefficient[1].y*circlePointPower1.y+pFourierCoefficient[2].x*circlePointPower2.x-pFourierCoefficient[2].y*circlePointPower2.y+pFourierCoefficient[3].x*circlePointPower3.x-pFourierCoefficient[3].y*circlePointPower3.y;
	return 2.0f*result+pFourierCoefficient[0].x;
}

FUNC float evaluateFourierSeries10(const float_complex circlePoint,const float_complex pFourierCoefficient[10]){
	// Minimize sequential dependencies by computing powers of circlePoint
	// by multiplying previous powers by powers of two
	float_complex circlePointPower1=circlePoint;
	float_complex circlePointPower2=circlePointPower1*circlePointPower1;
	float_complex circlePointPower3=circlePointPower1*circlePointPower2;
	float_complex circlePointPower4=circlePointPower2*circlePointPower2;
	float_complex circlePointPower5=circlePointPower1*circlePointPower4;
	float_complex circlePointPower6=circlePointPower2*circlePointPower4;
	float_complex circlePointPower7=circlePointPower3*circlePointPower4;
	float_complex circlePointPower8=circlePointPower4*circlePointPower4;
	float_complex circlePointPower9=circlePointPower1*circlePointPower8;
	float result=pFourierCoefficient[1].x*circlePointPower1.x-pFourierCoefficient[1].y*circlePointPower1.y+pFourierCoefficient[2].x*circlePointPower2.x-pFourierCoefficient[2].y*circlePointPower2.y+pFourierCoefficient[3].x*circlePointPower3.x-pFourierCoefficient[3].y*circlePointPower3.y+pFourierCoefficient[4].x*circlePointPower4.x-pFourierCoefficient[4].y*circlePointPower4.y+pFourierCoefficient[5].x*circlePointPower5.x-pFourierCoefficient[5].y*circlePointPower5.y+pFourierCoefficient[6].x*circlePointPower6.x-pFourierCoefficient[6].y*circlePointPower6.y+pFourierCoefficient[7].x*circlePointPower7.x-pFourierCoefficient[7].y*circlePointPower7.y+pFourierCoefficient[8].x*circlePointPower8.x-pFourierCoefficient[8].y*circlePointPower8.y+pFourierCoefficient[9].x*circlePointPower9.x-pFourierCoefficient[9].y*circlePointPower9.y;
	return 2.0f*result+pFourierCoefficient[0].x;
}


FUNC float evaluateFourierSeries4(const float_complex circlePoint,const float pFourierCoefficient[4]){
	// Minimize sequential dependencies by computing powers of circlePoint
	// by multiplying previous powers by powers of two
	float_complex circlePointPower1=circlePoint;
	float_complex circlePointPower2=circlePointPower1*circlePointPower1;
	float_complex circlePointPower3=circlePointPower1*circlePointPower2;
	float result=pFourierCoefficient[1]*circlePointPower1.x+pFourierCoefficient[2]*circlePointPower2.x+pFourierCoefficient[3]*circlePointPower3.x;
	return 2.0f*result+pFourierCoefficient[0];
}

FUNC float evaluateFourierSeries10(const float_complex circlePoint,const float pFourierCoefficient[10]){
	// Minimize sequential dependencies by computing powers of circlePoint
	// by multiplying previous powers by powers of two
	float_complex circlePointPower1=circlePoint;
	float_complex circlePointPower2=circlePointPower1*circlePointPower1;
	float_complex circlePointPower3=circlePointPower1*circlePointPower2;
	float_complex circlePointPower4=circlePointPower2*circlePointPower2;
	float_complex circlePointPower5=circlePointPower1*circlePointPower4;
	float_complex circlePointPower6=circlePointPower2*circlePointPower4;
	float_complex circlePointPower7=circlePointPower3*circlePointPower4;
	float_complex circlePointPower8=circlePointPower4*circlePointPower4;
	float_complex circlePointPower9=circlePointPower1*circlePointPower8;
	float result=pFourierCoefficient[1]*circlePointPower1.x+pFourierCoefficient[2]*circlePointPower2.x+pFourierCoefficient[3]*circlePointPower3.x+pFourierCoefficient[4]*circlePointPower4.x+pFourierCoefficient[5]*circlePointPower5.x+pFourierCoefficient[6]*circlePointPower6.x+pFourierCoefficient[7]*circlePointPower7.x+pFourierCoefficient[8]*circlePointPower8.x+pFourierCoefficient[9]*circlePointPower9.x;
	return 2.0f*result+pFourierCoefficient[0];
}


FUNC void prepareReflectanceSpectrum4(float_complex pOutExponentialMoment[4],float_complex pOutEvaluationPolynomial[4],const float_complex pTrigonometricMoment[4]){
	trigonometricToExponentialMoments4(pOutExponentialMoment,pTrigonometricMoment);
	levinsonsAlgorithm4(pOutEvaluationPolynomial,pOutExponentialMoment);
	pOutEvaluationPolynomial[0]=6.28318531f*pOutEvaluationPolynomial[0];
	pOutEvaluationPolynomial[1]=6.28318531f*pOutEvaluationPolynomial[1];
	pOutEvaluationPolynomial[2]=6.28318531f*pOutEvaluationPolynomial[2];
	pOutEvaluationPolynomial[3]=6.28318531f*pOutEvaluationPolynomial[3];
}

FUNC void prepareReflectanceSpectrumWithBiasing4(float_complex pOutExponentialMoment[4],float_complex pOutEvaluationPolynomial[4],float_complex pTrigonometricMoment[4]){
	pTrigonometricMoment[0]=float_complex((pTrigonometricMoment[0].x<0.0001f)?0.0001f:((pTrigonometricMoment[0].x>0.9999f)?0.9999f:pTrigonometricMoment[0].x),0.0f);
	trigonometricToExponentialMoments4(pOutExponentialMoment,pTrigonometricMoment);
	levinsonsAlgorithmWithBiasing4(pOutEvaluationPolynomial,pOutExponentialMoment);
	pOutEvaluationPolynomial[0]=6.28318531f*pOutEvaluationPolynomial[0];
	pOutEvaluationPolynomial[1]=6.28318531f*pOutEvaluationPolynomial[1];
	pOutEvaluationPolynomial[2]=6.28318531f*pOutEvaluationPolynomial[2];
	pOutEvaluationPolynomial[3]=6.28318531f*pOutEvaluationPolynomial[3];
}

FUNC void prepareReflectanceSpectrum4(float_complex pOutExponentialMoment[4],float_complex pOutEvaluationPolynomial[4],const float pTrigonometricMoment[4]){
	trigonometricToExponentialMoments4(pOutExponentialMoment,pTrigonometricMoment);
	levinsonsAlgorithm4(pOutEvaluationPolynomial,pOutExponentialMoment);
	pOutEvaluationPolynomial[0]=6.28318531f*pOutEvaluationPolynomial[0];
	pOutEvaluationPolynomial[1]=6.28318531f*pOutEvaluationPolynomial[1];
	pOutEvaluationPolynomial[2]=6.28318531f*pOutEvaluationPolynomial[2];
	pOutEvaluationPolynomial[3]=6.28318531f*pOutEvaluationPolynomial[3];
}

FUNC void prepareReflectanceSpectrumWithBiasing4(float_complex pOutExponentialMoment[4],float_complex pOutEvaluationPolynomial[4],float pTrigonometricMoment[4]){
	pTrigonometricMoment[0]=(pTrigonometricMoment[0]<0.0001f)?0.0001f:((pTrigonometricMoment[0]>0.9999f)?0.9999f:pTrigonometricMoment[0]);
	trigonometricToExponentialMoments4(pOutExponentialMoment,pTrigonometricMoment);
	levinsonsAlgorithmWithBiasing4(pOutEvaluationPolynomial,pOutExponentialMoment);
	pOutEvaluationPolynomial[0]=6.28318531f*pOutEvaluationPolynomial[0];
	pOutEvaluationPolynomial[1]=6.28318531f*pOutEvaluationPolynomial[1];
	pOutEvaluationPolynomial[2]=6.28318531f*pOutEvaluationPolynomial[2];
	pOutEvaluationPolynomial[3]=6.28318531f*pOutEvaluationPolynomial[3];
}

FUNC void prepareReflectanceSpectrum10(float_complex pOutExponentialMoment[10],float_complex pOutEvaluationPolynomial[10],const float_complex pTrigonometricMoment[10]){
	trigonometricToExponentialMoments10(pOutExponentialMoment,pTrigonometricMoment);
	levinsonsAlgorithm10(pOutEvaluationPolynomial,pOutExponentialMoment);
	pOutEvaluationPolynomial[0]=6.28318531f*pOutEvaluationPolynomial[0];
	pOutEvaluationPolynomial[1]=6.28318531f*pOutEvaluationPolynomial[1];
	pOutEvaluationPolynomial[2]=6.28318531f*pOutEvaluationPolynomial[2];
	pOutEvaluationPolynomial[3]=6.28318531f*pOutEvaluationPolynomial[3];
	pOutEvaluationPolynomial[4]=6.28318531f*pOutEvaluationPolynomial[4];
	pOutEvaluationPolynomial[5]=6.28318531f*pOutEvaluationPolynomial[5];
	pOutEvaluationPolynomial[6]=6.28318531f*pOutEvaluationPolynomial[6];
	pOutEvaluationPolynomial[7]=6.28318531f*pOutEvaluationPolynomial[7];
	pOutEvaluationPolynomial[8]=6.28318531f*pOutEvaluationPolynomial[8];
	pOutEvaluationPolynomial[9]=6.28318531f*pOutEvaluationPolynomial[9];
}

FUNC void prepareReflectanceSpectrumWithBiasing10(float_complex pOutExponentialMoment[10],float_complex pOutEvaluationPolynomial[10],float_complex pTrigonometricMoment[10]){
	pTrigonometricMoment[0]=float_complex((pTrigonometricMoment[0].x<0.0001f)?0.0001f:((pTrigonometricMoment[0].x>0.9999f)?0.9999f:pTrigonometricMoment[0].x),0.0f);
	trigonometricToExponentialMoments10(pOutExponentialMoment,pTrigonometricMoment);
	levinsonsAlgorithmWithBiasing10(pOutEvaluationPolynomial,pOutExponentialMoment);
	pOutEvaluationPolynomial[0]=6.28318531f*pOutEvaluationPolynomial[0];
	pOutEvaluationPolynomial[1]=6.28318531f*pOutEvaluationPolynomial[1];
	pOutEvaluationPolynomial[2]=6.28318531f*pOutEvaluationPolynomial[2];
	pOutEvaluationPolynomial[3]=6.28318531f*pOutEvaluationPolynomial[3];
	pOutEvaluationPolynomial[4]=6.28318531f*pOutEvaluationPolynomial[4];
	pOutEvaluationPolynomial[5]=6.28318531f*pOutEvaluationPolynomial[5];
	pOutEvaluationPolynomial[6]=6.28318531f*pOutEvaluationPolynomial[6];
	pOutEvaluationPolynomial[7]=6.28318531f*pOutEvaluationPolynomial[7];
	pOutEvaluationPolynomial[8]=6.28318531f*pOutEvaluationPolynomial[8];
	pOutEvaluationPolynomial[9]=6.28318531f*pOutEvaluationPolynomial[9];
}

FUNC void prepareReflectanceSpectrum10(float_complex pOutExponentialMoment[10],float_complex pOutEvaluationPolynomial[10],const float pTrigonometricMoment[10]){
	trigonometricToExponentialMoments10(pOutExponentialMoment,pTrigonometricMoment);
	levinsonsAlgorithm10(pOutEvaluationPolynomial,pOutExponentialMoment);
	pOutEvaluationPolynomial[0]=6.28318531f*pOutEvaluationPolynomial[0];
	pOutEvaluationPolynomial[1]=6.28318531f*pOutEvaluationPolynomial[1];
	pOutEvaluationPolynomial[2]=6.28318531f*pOutEvaluationPolynomial[2];
	pOutEvaluationPolynomial[3]=6.28318531f*pOutEvaluationPolynomial[3];
	pOutEvaluationPolynomial[4]=6.28318531f*pOutEvaluationPolynomial[4];
	pOutEvaluationPolynomial[5]=6.28318531f*pOutEvaluationPolynomial[5];
	pOutEvaluationPolynomial[6]=6.28318531f*pOutEvaluationPolynomial[6];
	pOutEvaluationPolynomial[7]=6.28318531f*pOutEvaluationPolynomial[7];
	pOutEvaluationPolynomial[8]=6.28318531f*pOutEvaluationPolynomial[8];
	pOutEvaluationPolynomial[9]=6.28318531f*pOutEvaluationPolynomial[9];
}

FUNC void prepareReflectanceSpectrumWithBiasing10(float_complex pOutExponentialMoment[10],float_complex pOutEvaluationPolynomial[10],float pTrigonometricMoment[10]){
	pTrigonometricMoment[0]=(pTrigonometricMoment[0]<0.0001f)?0.0001f:((pTrigonometricMoment[0]>0.9999f)?0.9999f:pTrigonometricMoment[0]);
	trigonometricToExponentialMoments10(pOutExponentialMoment,pTrigonometricMoment);
	levinsonsAlgorithmWithBiasing10(pOutEvaluationPolynomial,pOutExponentialMoment);
	pOutEvaluationPolynomial[0]=6.28318531f*pOutEvaluationPolynomial[0];
	pOutEvaluationPolynomial[1]=6.28318531f*pOutEvaluationPolynomial[1];
	pOutEvaluationPolynomial[2]=6.28318531f*pOutEvaluationPolynomial[2];
	pOutEvaluationPolynomial[3]=6.28318531f*pOutEvaluationPolynomial[3];
	pOutEvaluationPolynomial[4]=6.28318531f*pOutEvaluationPolynomial[4];
	pOutEvaluationPolynomial[5]=6.28318531f*pOutEvaluationPolynomial[5];
	pOutEvaluationPolynomial[6]=6.28318531f*pOutEvaluationPolynomial[6];
	pOutEvaluationPolynomial[7]=6.28318531f*pOutEvaluationPolynomial[7];
	pOutEvaluationPolynomial[8]=6.28318531f*pOutEvaluationPolynomial[8];
	pOutEvaluationPolynomial[9]=6.28318531f*pOutEvaluationPolynomial[9];
}


FUNC float evaluateReflectanceSpectrum4(const float phase,const float_complex pExponentialMoment[4],const float_complex pEvaluationPolynomial[4]){
	float_complex circlePoint;
	circlePoint=float_complex(cosf(phase),sinf(phase));
	float_complex herglotzTransform;
	herglotzTransform=evaluateFastHerglotzTransform4(circlePoint,pExponentialMoment,pEvaluationPolynomial);
	return fast_atan(herglotzTransform.y/herglotzTransform.x)*0.318309886f+0.5f;
}

FUNC float evaluateReflectanceSpectrum10(const float phase,const float_complex pExponentialMoment[10],const float_complex pEvaluationPolynomial[10]){
	float_complex circlePoint;
	circlePoint=float_complex(cosf(phase),sinf(phase));
	float_complex herglotzTransform;
	herglotzTransform=evaluateFastHerglotzTransform10(circlePoint,pExponentialMoment,pEvaluationPolynomial);
	return fast_atan(herglotzTransform.y/herglotzTransform.x)*0.318309886f+0.5f;
}


FUNC void prepareReflectanceSpectrumLagrange4(float_complex pOutLagrangeMultiplier[4],const float_complex pTrigonometricMoment[4]){
	float_complex pExponentialMoment[4];
	trigonometricToExponentialMoments4(pExponentialMoment,pTrigonometricMoment);
	float_complex pEvaluationPolynomial[4];
	levinsonsAlgorithm4(pEvaluationPolynomial,pExponentialMoment);
	pEvaluationPolynomial[0]=6.28318531f*pEvaluationPolynomial[0];
	pEvaluationPolynomial[1]=6.28318531f*pEvaluationPolynomial[1];
	pEvaluationPolynomial[2]=6.28318531f*pEvaluationPolynomial[2];
	pEvaluationPolynomial[3]=6.28318531f*pEvaluationPolynomial[3];
	float_complex pAutocorrelation[4];
	computeAutocorrelation4(pAutocorrelation,pEvaluationPolynomial);
	pExponentialMoment[0]=0.5f*pExponentialMoment[0];
	computeCorrelation4(pOutLagrangeMultiplier,pAutocorrelation,pExponentialMoment);
	float normalizationFactor=1.0f/(3.14159265f*pEvaluationPolynomial[0].x);
	pOutLagrangeMultiplier[0]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[0]);
	pOutLagrangeMultiplier[1]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[1]);
	pOutLagrangeMultiplier[2]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[2]);
	pOutLagrangeMultiplier[3]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[3]);
	pOutLagrangeMultiplier[0]=float_complex(pOutLagrangeMultiplier[0].x,0.0f);
}

FUNC void prepareReflectanceSpectrumLagrangeWithBiasing4(float_complex pOutLagrangeMultiplier[4],float_complex pTrigonometricMoment[4]){
	pTrigonometricMoment[0]=float_complex((pTrigonometricMoment[0].x<0.0001f)?0.0001f:((pTrigonometricMoment[0].x>0.9999f)?0.9999f:pTrigonometricMoment[0].x),0.0f);
	float_complex pExponentialMoment[4];
	trigonometricToExponentialMoments4(pExponentialMoment,pTrigonometricMoment);
	float_complex pEvaluationPolynomial[4];
	levinsonsAlgorithmWithBiasing4(pEvaluationPolynomial,pExponentialMoment);
	pEvaluationPolynomial[0]=6.28318531f*pEvaluationPolynomial[0];
	pEvaluationPolynomial[1]=6.28318531f*pEvaluationPolynomial[1];
	pEvaluationPolynomial[2]=6.28318531f*pEvaluationPolynomial[2];
	pEvaluationPolynomial[3]=6.28318531f*pEvaluationPolynomial[3];
	float_complex pAutocorrelation[4];
	computeAutocorrelation4(pAutocorrelation,pEvaluationPolynomial);
	pExponentialMoment[0]=0.5f*pExponentialMoment[0];
	computeCorrelation4(pOutLagrangeMultiplier,pAutocorrelation,pExponentialMoment);
	float normalizationFactor=1.0f/(3.14159265f*pEvaluationPolynomial[0].x);
	pOutLagrangeMultiplier[0]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[0]);
	pOutLagrangeMultiplier[1]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[1]);
	pOutLagrangeMultiplier[2]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[2]);
	pOutLagrangeMultiplier[3]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[3]);
	pOutLagrangeMultiplier[0]=float_complex(pOutLagrangeMultiplier[0].x,0.0f);
}

FUNC void prepareReflectanceSpectrumLagrange4(float pOutLagrangeMultiplier[4],const float pTrigonometricMoment[4]){
	float_complex pExponentialMoment[4];
	trigonometricToExponentialMoments4(pExponentialMoment,pTrigonometricMoment);
	float_complex pEvaluationPolynomial[4];
	levinsonsAlgorithm4(pEvaluationPolynomial,pExponentialMoment);
	pEvaluationPolynomial[0]=6.28318531f*pEvaluationPolynomial[0];
	pEvaluationPolynomial[1]=6.28318531f*pEvaluationPolynomial[1];
	pEvaluationPolynomial[2]=6.28318531f*pEvaluationPolynomial[2];
	pEvaluationPolynomial[3]=6.28318531f*pEvaluationPolynomial[3];
	float_complex pAutocorrelation[4];
	computeAutocorrelation4(pAutocorrelation,pEvaluationPolynomial);
	pExponentialMoment[0]=0.5f*pExponentialMoment[0];
	computeImaginaryCorrelation4(pOutLagrangeMultiplier,pAutocorrelation,pExponentialMoment);
	float normalizationFactor=1.0f/(3.14159265f*pEvaluationPolynomial[0].x);
	pOutLagrangeMultiplier[0]=normalizationFactor*pOutLagrangeMultiplier[0];
	pOutLagrangeMultiplier[1]=normalizationFactor*pOutLagrangeMultiplier[1];
	pOutLagrangeMultiplier[2]=normalizationFactor*pOutLagrangeMultiplier[2];
	pOutLagrangeMultiplier[3]=normalizationFactor*pOutLagrangeMultiplier[3];
}

FUNC void prepareReflectanceSpectrumLagrangeWithBiasing4(float pOutLagrangeMultiplier[4],float pTrigonometricMoment[4]){
	pTrigonometricMoment[0]=(pTrigonometricMoment[0]<0.0001f)?0.0001f:((pTrigonometricMoment[0]>0.9999f)?0.9999f:pTrigonometricMoment[0]);
	float_complex pExponentialMoment[4];
	trigonometricToExponentialMoments4(pExponentialMoment,pTrigonometricMoment);
	float_complex pEvaluationPolynomial[4];
	levinsonsAlgorithmWithBiasing4(pEvaluationPolynomial,pExponentialMoment);
	pEvaluationPolynomial[0]=6.28318531f*pEvaluationPolynomial[0];
	pEvaluationPolynomial[1]=6.28318531f*pEvaluationPolynomial[1];
	pEvaluationPolynomial[2]=6.28318531f*pEvaluationPolynomial[2];
	pEvaluationPolynomial[3]=6.28318531f*pEvaluationPolynomial[3];
	float_complex pAutocorrelation[4];
	computeAutocorrelation4(pAutocorrelation,pEvaluationPolynomial);
	pExponentialMoment[0]=0.5f*pExponentialMoment[0];
	computeImaginaryCorrelation4(pOutLagrangeMultiplier,pAutocorrelation,pExponentialMoment);
	float normalizationFactor=1.0f/(3.14159265f*pEvaluationPolynomial[0].x);
	pOutLagrangeMultiplier[0]=normalizationFactor*pOutLagrangeMultiplier[0];
	pOutLagrangeMultiplier[1]=normalizationFactor*pOutLagrangeMultiplier[1];
	pOutLagrangeMultiplier[2]=normalizationFactor*pOutLagrangeMultiplier[2];
	pOutLagrangeMultiplier[3]=normalizationFactor*pOutLagrangeMultiplier[3];
}

FUNC void prepareReflectanceSpectrumLagrange10(float_complex pOutLagrangeMultiplier[10],const float_complex pTrigonometricMoment[10]){
	float_complex pExponentialMoment[10];
	trigonometricToExponentialMoments10(pExponentialMoment,pTrigonometricMoment);
	float_complex pEvaluationPolynomial[10];
	levinsonsAlgorithm10(pEvaluationPolynomial,pExponentialMoment);
	pEvaluationPolynomial[0]=6.28318531f*pEvaluationPolynomial[0];
	pEvaluationPolynomial[1]=6.28318531f*pEvaluationPolynomial[1];
	pEvaluationPolynomial[2]=6.28318531f*pEvaluationPolynomial[2];
	pEvaluationPolynomial[3]=6.28318531f*pEvaluationPolynomial[3];
	pEvaluationPolynomial[4]=6.28318531f*pEvaluationPolynomial[4];
	pEvaluationPolynomial[5]=6.28318531f*pEvaluationPolynomial[5];
	pEvaluationPolynomial[6]=6.28318531f*pEvaluationPolynomial[6];
	pEvaluationPolynomial[7]=6.28318531f*pEvaluationPolynomial[7];
	pEvaluationPolynomial[8]=6.28318531f*pEvaluationPolynomial[8];
	pEvaluationPolynomial[9]=6.28318531f*pEvaluationPolynomial[9];
	float_complex pAutocorrelation[10];
	computeAutocorrelation10(pAutocorrelation,pEvaluationPolynomial);
	pExponentialMoment[0]=0.5f*pExponentialMoment[0];
	computeCorrelation10(pOutLagrangeMultiplier,pAutocorrelation,pExponentialMoment);
	float normalizationFactor=1.0f/(3.14159265f*pEvaluationPolynomial[0].x);
	pOutLagrangeMultiplier[0]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[0]);
	pOutLagrangeMultiplier[1]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[1]);
	pOutLagrangeMultiplier[2]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[2]);
	pOutLagrangeMultiplier[3]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[3]);
	pOutLagrangeMultiplier[4]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[4]);
	pOutLagrangeMultiplier[5]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[5]);
	pOutLagrangeMultiplier[6]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[6]);
	pOutLagrangeMultiplier[7]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[7]);
	pOutLagrangeMultiplier[8]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[8]);
	pOutLagrangeMultiplier[9]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[9]);
	pOutLagrangeMultiplier[0]=float_complex(pOutLagrangeMultiplier[0].x,0.0f);
}

FUNC void prepareReflectanceSpectrumLagrangeWithBiasing10(float_complex pOutLagrangeMultiplier[10],float_complex pTrigonometricMoment[10]){
	pTrigonometricMoment[0]=float_complex((pTrigonometricMoment[0].x<0.0001f)?0.0001f:((pTrigonometricMoment[0].x>0.9999f)?0.9999f:pTrigonometricMoment[0].x),0.0f);
	float_complex pExponentialMoment[10];
	trigonometricToExponentialMoments10(pExponentialMoment,pTrigonometricMoment);
	float_complex pEvaluationPolynomial[10];
	levinsonsAlgorithmWithBiasing10(pEvaluationPolynomial,pExponentialMoment);
	pEvaluationPolynomial[0]=6.28318531f*pEvaluationPolynomial[0];
	pEvaluationPolynomial[1]=6.28318531f*pEvaluationPolynomial[1];
	pEvaluationPolynomial[2]=6.28318531f*pEvaluationPolynomial[2];
	pEvaluationPolynomial[3]=6.28318531f*pEvaluationPolynomial[3];
	pEvaluationPolynomial[4]=6.28318531f*pEvaluationPolynomial[4];
	pEvaluationPolynomial[5]=6.28318531f*pEvaluationPolynomial[5];
	pEvaluationPolynomial[6]=6.28318531f*pEvaluationPolynomial[6];
	pEvaluationPolynomial[7]=6.28318531f*pEvaluationPolynomial[7];
	pEvaluationPolynomial[8]=6.28318531f*pEvaluationPolynomial[8];
	pEvaluationPolynomial[9]=6.28318531f*pEvaluationPolynomial[9];
	float_complex pAutocorrelation[10];
	computeAutocorrelation10(pAutocorrelation,pEvaluationPolynomial);
	pExponentialMoment[0]=0.5f*pExponentialMoment[0];
	computeCorrelation10(pOutLagrangeMultiplier,pAutocorrelation,pExponentialMoment);
	float normalizationFactor=1.0f/(3.14159265f*pEvaluationPolynomial[0].x);
	pOutLagrangeMultiplier[0]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[0]);
	pOutLagrangeMultiplier[1]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[1]);
	pOutLagrangeMultiplier[2]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[2]);
	pOutLagrangeMultiplier[3]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[3]);
	pOutLagrangeMultiplier[4]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[4]);
	pOutLagrangeMultiplier[5]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[5]);
	pOutLagrangeMultiplier[6]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[6]);
	pOutLagrangeMultiplier[7]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[7]);
	pOutLagrangeMultiplier[8]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[8]);
	pOutLagrangeMultiplier[9]=normalizationFactor*(float_complex(-0.0f,-1.0f)*pOutLagrangeMultiplier[9]);
	pOutLagrangeMultiplier[0]=float_complex(pOutLagrangeMultiplier[0].x,0.0f);
}

FUNC void prepareReflectanceSpectrumLagrange10(float pOutLagrangeMultiplier[10],const float pTrigonometricMoment[10]){
	float_complex pExponentialMoment[10];
	trigonometricToExponentialMoments10(pExponentialMoment,pTrigonometricMoment);
	float_complex pEvaluationPolynomial[10];
	levinsonsAlgorithm10(pEvaluationPolynomial,pExponentialMoment);
	pEvaluationPolynomial[0]=6.28318531f*pEvaluationPolynomial[0];
	pEvaluationPolynomial[1]=6.28318531f*pEvaluationPolynomial[1];
	pEvaluationPolynomial[2]=6.28318531f*pEvaluationPolynomial[2];
	pEvaluationPolynomial[3]=6.28318531f*pEvaluationPolynomial[3];
	pEvaluationPolynomial[4]=6.28318531f*pEvaluationPolynomial[4];
	pEvaluationPolynomial[5]=6.28318531f*pEvaluationPolynomial[5];
	pEvaluationPolynomial[6]=6.28318531f*pEvaluationPolynomial[6];
	pEvaluationPolynomial[7]=6.28318531f*pEvaluationPolynomial[7];
	pEvaluationPolynomial[8]=6.28318531f*pEvaluationPolynomial[8];
	pEvaluationPolynomial[9]=6.28318531f*pEvaluationPolynomial[9];
	float_complex pAutocorrelation[10];
	computeAutocorrelation10(pAutocorrelation,pEvaluationPolynomial);
	pExponentialMoment[0]=0.5f*pExponentialMoment[0];
	computeImaginaryCorrelation10(pOutLagrangeMultiplier,pAutocorrelation,pExponentialMoment);
	float normalizationFactor=1.0f/(3.14159265f*pEvaluationPolynomial[0].x);
	pOutLagrangeMultiplier[0]=normalizationFactor*pOutLagrangeMultiplier[0];
	pOutLagrangeMultiplier[1]=normalizationFactor*pOutLagrangeMultiplier[1];
	pOutLagrangeMultiplier[2]=normalizationFactor*pOutLagrangeMultiplier[2];
	pOutLagrangeMultiplier[3]=normalizationFactor*pOutLagrangeMultiplier[3];
	pOutLagrangeMultiplier[4]=normalizationFactor*pOutLagrangeMultiplier[4];
	pOutLagrangeMultiplier[5]=normalizationFactor*pOutLagrangeMultiplier[5];
	pOutLagrangeMultiplier[6]=normalizationFactor*pOutLagrangeMultiplier[6];
	pOutLagrangeMultiplier[7]=normalizationFactor*pOutLagrangeMultiplier[7];
	pOutLagrangeMultiplier[8]=normalizationFactor*pOutLagrangeMultiplier[8];
	pOutLagrangeMultiplier[9]=normalizationFactor*pOutLagrangeMultiplier[9];
}

FUNC void prepareReflectanceSpectrumLagrangeWithBiasing10(float pOutLagrangeMultiplier[10],float pTrigonometricMoment[10]){
	pTrigonometricMoment[0]=(pTrigonometricMoment[0]<0.0001f)?0.0001f:((pTrigonometricMoment[0]>0.9999f)?0.9999f:pTrigonometricMoment[0]);
	float_complex pExponentialMoment[10];
	trigonometricToExponentialMoments10(pExponentialMoment,pTrigonometricMoment);
	float_complex pEvaluationPolynomial[10];
	levinsonsAlgorithmWithBiasing10(pEvaluationPolynomial,pExponentialMoment);
	pEvaluationPolynomial[0]=6.28318531f*pEvaluationPolynomial[0];
	pEvaluationPolynomial[1]=6.28318531f*pEvaluationPolynomial[1];
	pEvaluationPolynomial[2]=6.28318531f*pEvaluationPolynomial[2];
	pEvaluationPolynomial[3]=6.28318531f*pEvaluationPolynomial[3];
	pEvaluationPolynomial[4]=6.28318531f*pEvaluationPolynomial[4];
	pEvaluationPolynomial[5]=6.28318531f*pEvaluationPolynomial[5];
	pEvaluationPolynomial[6]=6.28318531f*pEvaluationPolynomial[6];
	pEvaluationPolynomial[7]=6.28318531f*pEvaluationPolynomial[7];
	pEvaluationPolynomial[8]=6.28318531f*pEvaluationPolynomial[8];
	pEvaluationPolynomial[9]=6.28318531f*pEvaluationPolynomial[9];
	float_complex pAutocorrelation[10];
	computeAutocorrelation10(pAutocorrelation,pEvaluationPolynomial);
	pExponentialMoment[0]=0.5f*pExponentialMoment[0];
	computeImaginaryCorrelation10(pOutLagrangeMultiplier,pAutocorrelation,pExponentialMoment);
	float normalizationFactor=1.0f/(3.14159265f*pEvaluationPolynomial[0].x);
	pOutLagrangeMultiplier[0]=normalizationFactor*pOutLagrangeMultiplier[0];
	pOutLagrangeMultiplier[1]=normalizationFactor*pOutLagrangeMultiplier[1];
	pOutLagrangeMultiplier[2]=normalizationFactor*pOutLagrangeMultiplier[2];
	pOutLagrangeMultiplier[3]=normalizationFactor*pOutLagrangeMultiplier[3];
	pOutLagrangeMultiplier[4]=normalizationFactor*pOutLagrangeMultiplier[4];
	pOutLagrangeMultiplier[5]=normalizationFactor*pOutLagrangeMultiplier[5];
	pOutLagrangeMultiplier[6]=normalizationFactor*pOutLagrangeMultiplier[6];
	pOutLagrangeMultiplier[7]=normalizationFactor*pOutLagrangeMultiplier[7];
	pOutLagrangeMultiplier[8]=normalizationFactor*pOutLagrangeMultiplier[8];
	pOutLagrangeMultiplier[9]=normalizationFactor*pOutLagrangeMultiplier[9];
}


FUNC float evaluateReflectanceSpectrumLagrange4(const float phase,const float_complex pLagrangeMultiplier[4]){
	float_complex conjCirclePoint;
	conjCirclePoint=float_complex(cosf(-phase),sinf(-phase));
	float lagrangeSeries;
	lagrangeSeries=evaluateFourierSeries4(conjCirclePoint,pLagrangeMultiplier);
	return fast_atan(lagrangeSeries)*0.318309886f+0.5f;
}

FUNC float evaluateReflectanceSpectrumLagrange4(const float phase,const float pLagrangeMultiplier[4]){
	float_complex conjCirclePoint;
	conjCirclePoint=float_complex(cosf(-phase),sinf(-phase));
	float lagrangeSeries;
	lagrangeSeries=evaluateFourierSeries4(conjCirclePoint,pLagrangeMultiplier);
	return fast_atan(lagrangeSeries)*0.318309886f+0.5f;
}

FUNC float evaluateReflectanceSpectrumLagrange10(const float phase,const float_complex pLagrangeMultiplier[10]){
	float_complex conjCirclePoint;
	conjCirclePoint=float_complex(cosf(-phase),sinf(-phase));
	float lagrangeSeries;
	lagrangeSeries=evaluateFourierSeries10(conjCirclePoint,pLagrangeMultiplier);
	return fast_atan(lagrangeSeries)*0.318309886f+0.5f;
}

FUNC float evaluateReflectanceSpectrumLagrange10(const float phase,const float pLagrangeMultiplier[10]){
	float_complex conjCirclePoint;
	conjCirclePoint=float_complex(cosf(-phase),sinf(-phase));
	float lagrangeSeries;
	lagrangeSeries=evaluateFourierSeries10(conjCirclePoint,pLagrangeMultiplier);
	return fast_atan(lagrangeSeries)*0.318309886f+0.5f;
}


