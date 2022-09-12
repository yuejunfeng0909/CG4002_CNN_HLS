#include "fir_cnn.h"

CNN_IN_DTYPE input_buffer[CNN_KERNEL_LENGTH][INPUT_DEPTH];
CNN_OUT_DTYPE cnn_output_buffer[OUTPUT_LENGTH][OUTPUT_DEPTH];

void read_input(CNN_RAW_IN_DTYPE *input){
#pragma HLS ARRAY_PARTITION variable=input type=complete
	// shift input register
	for (int d = 0; d < INPUT_DEPTH; d++) {
#pragma HLS UNROLL
		for (int i = 1; i < CNN_KERNEL_LENGTH; i++) {
			input_buffer[i][d] = input_buffer[i-1][d];
		}
		input_buffer[0][d] = (CNN_IN_DTYPE)input[d];
	}
}

void reset() {
	for (int i = 0; i < OUTPUT_LENGTH; i++) {
#pragma HLS UNROLL
			for (int d = 0; d < OUTPUT_DEPTH; d++) {
				cnn_output_buffer[i][d] = 0;
			}
	}
}

void compute_convolution() {
	// shift output register
	for (int i = 1; i < OUTPUT_LENGTH; i++) {
#pragma HLS UNROLL
		for (int j = 0; j < OUTPUT_DEPTH; j++) {
			cnn_output_buffer[i][j] = cnn_output_buffer[i- 1][j];
		}
	}

	// for each filter == output depth
	for (int depth = 0; depth < CNN_KERNEL_COUNT; depth++) {
#pragma HLS PIPELINE
		// for each data point
		CNN_OUT_DTYPE Oi = 0;
		for (int l = 0; l < CNN_KERNEL_LENGTH; l++) {
#pragma HLS UNROLL
			// for each value in the depth
			for (int d = 0; d < CNN_KERNEL_DEPTH; d++) {
				Oi += input_buffer[l][d] * CNN_weights[depth][CNN_KERNEL_LENGTH - l - 1][d];
			}
		}
		cnn_output_buffer[0][depth] = relu<CNN_OUT_DTYPE>(Oi + CNN_bias[depth]);
	}
}
