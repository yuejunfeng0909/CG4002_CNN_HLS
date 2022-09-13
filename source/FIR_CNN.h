#ifndef FIR_CNN
#define FIR_CNN

#include "set_weight_bias.h"
#include "activation.h"

typedef float CNN_RAW_IN_DTYPE;

void read_input(
		CNN_RAW_IN_DTYPE input[INPUT_DEPTH],
		CNN_IN_DTYPE input_buffer[CNN_KERNEL_LENGTH][INPUT_DEPTH]);
void reset(CNN_OUT_DTYPE cnn_output_buffer[CNN_OUTPUT_LENGTH][CNN_OUTPUT_DEPTH]);
void compute_convolution(
		CNN_IN_DTYPE input_buffer[CNN_KERNEL_LENGTH][INPUT_DEPTH],
		CNN_WEIGHTS_DTYPE CNN_weights[CNN_KERNEL_LENGTH][CNN_KERNEL_DEPTH][CNN_KERNEL_COUNT],
		CNN_BIAS_DTYPE CNN_bias[CNN_KERNEL_COUNT],
		CNN_OUT_DTYPE cnn_output_buffer[CNN_OUTPUT_LENGTH][CNN_OUTPUT_DEPTH]);

#endif
