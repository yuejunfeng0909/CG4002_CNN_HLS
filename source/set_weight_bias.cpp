#include "set_weight_bias.h"

// void set_CNN_weights_and_bias(IN_CNN_WEIGHTS_DTYPE *in_weights, IN_CNN_BIAS_DTYPE *in_bias) {
// 	CNN_WEIGHTS_DTYPE *cnn_weights_alis = &CNN_weights[0][0][0];
// 	copy<IN_CNN_WEIGHTS_DTYPE, CNN_WEIGHTS_DTYPE>(in_weights, cnn_weights_alis, CNN_KERNEL_COUNT * CNN_KERNEL_LENGTH * CNN_KERNEL_DEPTH);
// 	copy<IN_CNN_BIAS_DTYPE, CNN_BIAS_DTYPE>(in_bias, CNN_bias, CNN_KERNEL_COUNT);
// }

// void set_dense_weights_and_bias(IN_DENSE_WEIGHTS_DTYPE *weights, IN_DENSE_BIAS_DTYPE *bias) {
// 	DENSE_WEIGHTS_DTYPE *dense_weights_alis = &dense_weights[0][0];
// 	copy<IN_DENSE_WEIGHTS_DTYPE, DENSE_WEIGHTS_DTYPE>(weights, dense_weights_alis, DENSE_OUTPUT_NODES * DENSE_INPUT_NODES);
// 	copy<IN_DENSE_BIAS_DTYPE, DENSE_BIAS_DTYPE>(bias, dense_bias, DENSE_OUTPUT_NODES);
// }
