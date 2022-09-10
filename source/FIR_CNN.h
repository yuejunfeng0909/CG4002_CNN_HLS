#ifndef FIR_CNN
#define FIR_CNN

#include <ap_fixed.h>
#include "set_weight_bias.h"
#include "activation.h"

#define INPUT_DEPTH 6
#define INPUT_LENGTH 75

#define CNN_KERNEL_DEPTH 6
#define CNN_KERNEL_LENGTH 25
#define CNN_KERNEL_STRIDE 5
#define CNN_KERNEL_COUNT 16

#define OUTPUT_DEPTH CNN_KERNEL_COUNT
#define OUTPUT_LENGTH (INPUT_LENGTH - CNN_KERNEL_LENGTH) / CNN_KERNEL_STRIDE + 1

CNN_IN_DTYPE input_buffer[CNN_KERNEL_LENGTH][INPUT_DEPTH];
#pragma HLS ARRAY_PARTITION variable=input_buffer complete

CNN_OUT_DTYPE output_buffer[OUTPUT_LENGTH][OUTPUT_DEPTH];
#pragma HLS ARRAY_PARTITION variable=output_buffer complete

void read_input(CNN_IN_DTYPE input[input_depth]);
void reset();
void compute_convolution(CNN_IN_DTYPE input[input_depth]);



#endif
