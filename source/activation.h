/*
 * Rectified Linear Unit
 * Created: 4 Sep 2022
 * Author: Yue Junfeng
 */

#ifndef ACTIVATION
#define ACTIVATION

#include "hls_math.h"

template <typename DTYPE>
void relu(DTYPE x, DTYPE *y){
#pragma HLS INLINE
	*y = x > 0 ? x : 0;
}

template <typename INTYPE, typename OUTTYPE>
float softmax(INTYPE *x, OUTTYPE *y, int size){
#pragma HLS INLINE
	INTYPE sum = 0;
	INTYPE exponent_buffer[size];
#pragma HLS ARRAY_PARTITION variable=exponent_buffer type=complete

	for(int i = 0; i < size; i++){
#pragma HLS UNROLL
		exponent_buffer[i] = exp(x[i]);
		sum += exponent_buffer[i];
	}
	for(int i = 0; i < size; i++){
#pragma HLS UNROLL
		y[i] = exponent_buffer[i] / sum;
	}
}

#endif
