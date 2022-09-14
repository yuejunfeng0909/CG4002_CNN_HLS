#ifndef WEIGHTS_AND_BIAS
#define WEIGHTS_AND_BIAS

#include <stdio.h>

typedef float INPUT_DTYPE;
typedef INPUT_DTYPE IN_CNN_WEIGHTS_DTYPE;
typedef INPUT_DTYPE IN_CNN_BIAS_DTYPE;
typedef INPUT_DTYPE IN_DENSE_WEIGHTS_DTYPE;
typedef INPUT_DTYPE IN_DENSE_BIAS_DTYPE;

typedef float WEIGHT_BIAS_DTYPE;
typedef WEIGHT_BIAS_DTYPE CNN_WEIGHTS_DTYPE;
typedef WEIGHT_BIAS_DTYPE CNN_BIAS_DTYPE;
typedef WEIGHT_BIAS_DTYPE DENSE_WEIGHTS_DTYPE;
typedef WEIGHT_BIAS_DTYPE DENSE_BIAS_DTYPE;

typedef float CNN_DTYPE;
typedef CNN_DTYPE CNN_IN_DTYPE;
typedef CNN_DTYPE CNN_OUT_DTYPE;

#define INPUT_DEPTH 6
#define INPUT_LENGTH 75

#define CNN_OUTPUT_DEPTH CNN_KERNEL_COUNT
#define CNN_OUTPUT_LENGTH ((INPUT_LENGTH - CNN_KERNEL_LENGTH) / CNN_KERNEL_STRIDE + 1)

#define CNN_KERNEL_DEPTH 6
#define CNN_KERNEL_LENGTH 15
#define CNN_KERNEL_STRIDE 5
#define CNN_KERNEL_COUNT 10

#define DENSE_INPUT_NODES CNN_OUTPUT_LENGTH * CNN_KERNEL_COUNT
#define DENSE_OUTPUT_NODES 3

template <typename IN_TYPE, typename OUT_TYPE>
void copy(IN_TYPE *from, OUT_TYPE *to, int size) {
	for (int i = 0; i < size; i++) {
#pragma HLS UNROLL
		to[i] = from[i];
	}
}

template <typename IN_TYPE, typename OUT_TYPE>
void copy_inputs(IN_TYPE from[], OUT_TYPE to[]) {
	for (int i = 0; i < INPUT_LENGTH; i++) {
#pragma HLS UNROLL
		for (int j = 0; j < INPUT_DEPTH; j++) {
			to[i][j] = from[i][j]/4096.0f;
		}
	}
	printf("sample: %f should be ~120\n", to[0][0]);
}

// void set_CNN_weights_and_bias(
// 		IN_CNN_WEIGHTS_DTYPE in_weights[CNN_KERNEL_COUNT * CNN_KERNEL_LENGTH * CNN_KERNEL_DEPTH], 
// 		IN_CNN_BIAS_DTYPE in_bias[CNN_KERNEL_COUNT],
// 		CNN_WEIGHTS_DTYPE CNN_weights[CNN_KERNEL_COUNT][CNN_KERNEL_LENGTH][CNN_KERNEL_DEPTH],
// 		CNN_BIAS_DTYPE CNN_bias[CNN_KERNEL_COUNT]) {
// 	CNN_WEIGHTS_DTYPE *cnn_weights_alis = &CNN_weights[0][0][0];
// 	copy<IN_CNN_WEIGHTS_DTYPE, CNN_WEIGHTS_DTYPE>(in_weights, cnn_weights_alis, CNN_KERNEL_COUNT * CNN_KERNEL_LENGTH * CNN_KERNEL_DEPTH);
// 	copy<IN_CNN_BIAS_DTYPE, CNN_BIAS_DTYPE>(in_bias, CNN_bias, CNN_KERNEL_COUNT);
// }

// void set_dense_weights_and_bias(
// 		IN_DENSE_WEIGHTS_DTYPE weights[DENSE_OUTPUT_NODES * DENSE_INPUT_NODES], 
// 		IN_DENSE_BIAS_DTYPE bias[DENSE_OUTPUT_NODES],
// 		DENSE_WEIGHTS_DTYPE dense_weights[DENSE_OUTPUT_NODES][DENSE_INPUT_NODES],
// 		DENSE_BIAS_DTYPE dense_bias[DENSE_OUTPUT_NODES]) {
// 	DENSE_WEIGHTS_DTYPE *dense_weights_alis = &dense_weights[0][0];
// 	copy<IN_DENSE_WEIGHTS_DTYPE, DENSE_WEIGHTS_DTYPE>(weights, dense_weights_alis, DENSE_OUTPUT_NODES * DENSE_INPUT_NODES);
// 	copy<IN_DENSE_BIAS_DTYPE, DENSE_BIAS_DTYPE>(bias, dense_bias, DENSE_OUTPUT_NODES);
// }

#endif
