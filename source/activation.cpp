#include "activation.h"

//template <typename DTYPE>
//DTYPE relu(DTYPE x){
//#pragma HLS INLINE
//	return x > 0 ? x : 0;
//}

//template <typename INTYPE, typename OUTTYPE>
//void softmax(INTYPE x[], OUTTYPE y[], int size){
//#pragma HLS INLINE
//	OUTTYPE sum = 0;
//	OUTTYPE exponent_buffer[size];
//#pragma HLS ARRAY_PARTITION variable=exponent_buffer type=complete
//
//	for(int i = 0; i < size; i++){
//#pragma HLS UNROLL
//		exponent_buffer[i] = exp(x[i]);
//		sum += exponent_buffer[i];
//	}
//	for(int i = 0; i < size; i++){
//#pragma HLS UNROLL
//		y[i] = exponent_buffer[i] / sum;
//	}
//}
