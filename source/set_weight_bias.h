#ifndef WEIGHTS_AND_BIAS
#define WEIGHTS_AND_BIAS

#include <ap_fixed.h>

typedef float IN_CNN_WEIGHTS_DTYPE;
typedef IN_CNN_WEIGHTS_DTYPE IN_CNN_BIAS_DTYPE;
typedef ap_fixed<16, 1> CNN_WEIGHTS_DTYPE;
typedef CNN_WEIGHTS_DTYPE CNN_BIAS_DTYPE;

#define CNN_KERNEL_COUNT 16
#define CNN_KERNEL_LENGTH 10
#define CNN_KERNEL_STRIDE 5
#define CNN_KERNEL_DEPTH 6

CNN_WEIGHTS_DTYPE CNN_weights[CNN_KERNEL_COUNT][CNN_KERNEL_LENGTH][CNN_KERNEL_DEPTH];
CNN_BIAS_DTYPE CNN_bias[CNN_KERNEL_COUNT];

template <typename IN_TYPE, typename OUT_TYPE>
void copy(IN_TYPE *from, OUT_TYPE *to, int size);

void set_CNN_weights_and_bias(float *weights, float *bias);


#endif
