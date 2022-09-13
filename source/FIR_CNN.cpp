#include "fir_cnn.h"

void read_input(
		CNN_RAW_IN_DTYPE input[INPUT_DEPTH],
		CNN_IN_DTYPE input_buffer[CNN_KERNEL_LENGTH][INPUT_DEPTH]){
#pragma HLS ARRAY_PARTITION variable=input type=complete
#pragma HLS ARRAY_PARTITION variable=input_buffer type=cyclic dim=2 factor=2
	for (int d = 0; d < INPUT_DEPTH; d++) {
#pragma HLS unroll
		for (int i = 1; i < CNN_KERNEL_LENGTH; i++) {
			input_buffer[i][d] = input_buffer[i-1][d];
		}
		input_buffer[0][d] = (CNN_IN_DTYPE)input[d];
	}
}

void reset(CNN_OUT_DTYPE cnn_output_buffer[OUTPUT_LENGTH][OUTPUT_DEPTH]) {
#pragma HLS ARRAY_PARTITION variable=cnn_output_buffer type=cyclic dim=2 factor=2
	for (int i = 0; i < OUTPUT_LENGTH; i++) {
#pragma HLS unroll
			for (int d = 0; d < OUTPUT_DEPTH; d++) {
				cnn_output_buffer[i][d] = 0;
			}
	}
}

void compute_convolution(
		CNN_IN_DTYPE input_buffer[CNN_KERNEL_LENGTH][INPUT_DEPTH],
		CNN_WEIGHTS_DTYPE CNN_weights[CNN_KERNEL_COUNT][CNN_KERNEL_LENGTH][CNN_KERNEL_DEPTH],
		CNN_BIAS_DTYPE CNN_bias[CNN_KERNEL_COUNT],
		CNN_OUT_DTYPE cnn_output_buffer[OUTPUT_LENGTH][OUTPUT_DEPTH]) {
#pragma HLS ARRAY_PARTITION variable=input_buffer type=cyclic dim=2 factor=2
#pragma HLS ARRAY_PARTITION variable=CNN_weights type=cyclic dim=3 factor=2
#pragma HLS ARRAY_PARTITION variable=CNN_bias type=complete
#pragma HLS ARRAY_PARTITION variable=cnn_output_buffer type=cyclic dim=2 factor=2
	// shift output register
	for (int i = 1; i < OUTPUT_LENGTH; i++) {
		for (int j = 0; j < OUTPUT_DEPTH; j++) {
			cnn_output_buffer[i][j] = cnn_output_buffer[i- 1][j];
		}
	}

	// for each filter == output depth
	for (int depth = 0; depth < CNN_KERNEL_COUNT; depth++) {
#pragma HLS pipeline
		// for each data point
		CNN_OUT_DTYPE Oi = 0;
		for (int l = 0; l < CNN_KERNEL_LENGTH; l++) {
#pragma HLS unroll
			// for each value in the depth
			for (int d = 0; d < CNN_KERNEL_DEPTH; d++) {
				Oi += input_buffer[l][d] * CNN_weights[depth][CNN_KERNEL_LENGTH - l - 1][d];
			}
		}
		cnn_output_buffer[0][depth] = relu<CNN_OUT_DTYPE>(Oi + CNN_bias[depth]);
	}
}
