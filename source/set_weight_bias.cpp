#include "set_weight_bias.h"

template <typename IN_TYPE, typename OUT_TYPE>
void copy(IN_TYPE *from, OUT_TYPE *to, int size) {
#pragma HLS INLINE
	for (int i = 0; i < size; i++) {
		to[i] = from[i];
	}
}

void set_CNN_weights_and_bias(IN_CNN_WEIGHTS_DTYPE *in_weights, float *in_bias) {
	copy<IN_CNN_WEIGHTS_DTYPE, CNN_WEIGHTS_DTYPE>(in_weights, CNN_weights, CNN_KERNEL_COUNT * CNN_KERNEL_LENGTH * CNN_KERNEL_DEPTH);
	copy<IN_CNN_BIAS_DTYPE, CNN_BIAS_DTYPE>(in_bias, CNN_bias, CNN_KERNEL_COUNT);
}

void set_dense_weights_and_bias(float *weights, float *bias) {
	copy<IN_DENSE_WEIGHTS_DTYPE, DENSE_WEIGHTS_DTYPE>(weights, dense_weights, DENSE_OUTPUT_NODES * DENSE_INPUT_NODES)
	copy<IN_DENSE_BIAS_DTYPE, DENSE_BIAS_DTYPE>(bias, dense_bias, DENSE_OUTPUT_NODES);
}
