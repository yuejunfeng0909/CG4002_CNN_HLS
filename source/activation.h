/*
 * Rectified Linear Unit
 * Created: 4 Sep 2022
 * Author: Yue Junfeng
 */

#ifndef ACTIVATION
#define ACTIVATION

#include "model_definition.h"
#include "hls_math.h"


float relu(float x){
#pragma HLS INLINE
	return x > 0 ? x : 0;
}

float softmax(float x[OUTPUT_SIZE]){
#pragma HLS INLINE
	float sum = 0;
	for(int i = 0; i < OUTPUT_SIZE; i++){
#pragma HLS UNROLL
		sum += exp(x[i]);
	}
	for(int i = 0; i < OUTPUT_SIZE; i++){
#pragma HLS UNROLL
		x[i] = exp(x[i]) / sum;
	}
}

#endif
