#ifndef FIR_CNN
#define FIR_CNN

#include <ap_fixed.h>
#include "set_weight_bias.h"
#include "activation.h"

CNN_IN_DTYPE input_buffer[CNN_KERNEL_LENGTH][INPUT_DEPTH];
#pragma HLS ARRAY_PARTITION variable=input_buffer complete

CNN_OUT_DTYPE cnn_output_buffer[OUTPUT_LENGTH][OUTPUT_DEPTH];
#pragma HLS ARRAY_PARTITION variable=cnn_output_buffer complete

void read_input(CNN_IN_DTYPE input[input_depth]);
void reset();
void compute_convolution();



#endif
