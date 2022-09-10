#include "FIR_CNN.h"

void read_input(CNN_IN_DTYPE input[input_depth]){
#pragma HLS ARRAY_PARTITION variable=input type=complete
	// shift input register
	for (int d = 0; d < INPUT_DEPTH; d++) {
#pragma HLS UNROLL
		for (int i = 1; i < CNN_KERNEL_LENGTH; i++) {
			input_buffer[i][d] = input_buffer[i-1][d];
		}
		input_buffer[0][d] = input[d];
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
	depth: for (int depth = 0; depth < CNN_KERNEL_COUNT; depth++) {
#pragma HLS PIPELINE
		// for each data point
		CNN_OUT_DTYPE Oi = 0;
		row: for (int l = 0; l < CNN_KERNEL_LENGTH; l++) {
#pragma HLS UNROLL
			// for each value in the depth
			col: for (int d = 0; d < CNN_KERNEL_DEPTH; d++) {
				Oi += input_buffer[l][d] * CNN_weights[depth][CNN_KERNEL_LENGTH - l - 1][d];
			}
		}
		relu<CNN_OUT_DTYPE>(Oi + CNN_bias[depth], cnn_output_buffer[0][depth]);
	}
}
