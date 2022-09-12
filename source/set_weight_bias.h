#ifndef WEIGHTS_AND_BIAS
#define WEIGHTS_AND_BIAS

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

#define OUTPUT_DEPTH CNN_KERNEL_COUNT
#define OUTPUT_LENGTH ((INPUT_LENGTH - CNN_KERNEL_LENGTH) / CNN_KERNEL_STRIDE + 1)

#define CNN_KERNEL_DEPTH 6
#define CNN_KERNEL_LENGTH 25
#define CNN_KERNEL_STRIDE 5
#define CNN_KERNEL_COUNT 16

#define DENSE_INPUT_NODES OUTPUT_LENGTH * CNN_KERNEL_COUNT
#define DENSE_OUTPUT_NODES 2

extern CNN_WEIGHTS_DTYPE CNN_weights[CNN_KERNEL_COUNT][CNN_KERNEL_LENGTH][CNN_KERNEL_DEPTH];
extern CNN_BIAS_DTYPE CNN_bias[CNN_KERNEL_COUNT];

extern DENSE_WEIGHTS_DTYPE dense_weights[DENSE_OUTPUT_NODES][DENSE_INPUT_NODES];
extern DENSE_BIAS_DTYPE dense_bias[DENSE_OUTPUT_NODES];

template <typename IN_TYPE, typename OUT_TYPE>
void copy(IN_TYPE *from, OUT_TYPE *to, int size) {
#pragma HLS INLINE
	for (int i = 0; i < size; i++) {
		to[i] = from[i];
	}
}

// void set_CNN_weights_and_bias(float *weights, float *bias);
// void set_dense_weights_and_bias(float *weights, float *bias);

#endif