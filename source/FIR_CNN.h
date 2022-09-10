#ifndef FIR_CNN
#define FIR_CNN

#include "set_weight_bias.h"
#include "activation.h"

CNN_IN_DTYPE input_buffer[CNN_KERNEL_LENGTH][INPUT_DEPTH];

CNN_OUT_DTYPE cnn_output_buffer[OUTPUT_LENGTH][OUTPUT_DEPTH];

void read_input(CNN_IN_DTYPE *input);
void reset();
void compute_convolution();



#endif
