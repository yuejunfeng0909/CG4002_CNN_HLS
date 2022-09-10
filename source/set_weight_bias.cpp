#include "set_weight_bias.h"

template <typename IN_TYPE, typename OUT_TYPE>
void copy(IN_TYPE *from, OUT_TYPE *to, int size) {
#pragma HLS INLINE
	for (int i = 0; i < size; i++) {
		to[i] = from[i];
	}
}

void set_CNN_weights_and_bias(float *in_weights, float *in_bias) {
	copy<IN_CNN_WEIGHTS_DTYPE, CNN_WEIGHTS_DTYPE>(in_weights, CNN_weights, CNN_KERNEL_COUNT * CNN_KERNEL_LENGTH * CNN_KERNEL_DEPTH);
	copy<IN_CNN_BIAS_DTYPE, CNN_BIAS_DTYPE>(in_bias, CNN_bias, CNN_KERNEL_COUNT);
}
