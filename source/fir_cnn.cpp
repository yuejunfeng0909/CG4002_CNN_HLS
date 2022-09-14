#include "fir_cnn.h"
#include <stdio.h>

void reset(CNN_OUT_DTYPE cnn_output_buffer[CNN_OUTPUT_LENGTH][CNN_OUTPUT_DEPTH]) {
#pragma HLS ARRAY_PARTITION variable=cnn_output_buffer type=cyclic dim=1 complete
#pragma HLS INLINE
	CNN_OUTPUT_REG_RESET: for (int i = 0; i < CNN_OUTPUT_LENGTH; i++) {
// #pragma HLS unroll
			for (int d = 0; d < CNN_OUTPUT_DEPTH; d++) {
				cnn_output_buffer[i][d] = 0;
			}
	}
}

void compute_convolution(
		CNN_IN_DTYPE input_buffer[CNN_KERNEL_LENGTH][INPUT_DEPTH],
		CNN_WEIGHTS_DTYPE CNN_weights[CNN_KERNEL_LENGTH][CNN_KERNEL_DEPTH][CNN_KERNEL_COUNT],
		CNN_BIAS_DTYPE CNN_bias[CNN_KERNEL_COUNT],
		CNN_OUT_DTYPE cnn_output_buffer[CNN_OUTPUT_LENGTH][CNN_OUTPUT_DEPTH]) {
#pragma HLS ARRAY_PARTITION variable=input_buffer cyclic dim=2 factor=16
#pragma HLS ARRAY_PARTITION variable=CNN_weights cyclic dim=2 factor=16
#pragma HLS ARRAY_PARTITION variable=CNN_bias cyclic factor=16

	// shift output register
	CNN_REGISTER_SHIFT: for (int d = 0; d < CNN_OUTPUT_DEPTH; d++) {
#pragma HLS pipeline
		for (int i = 0; i < CNN_OUTPUT_LENGTH - 1; i++) {
			cnn_output_buffer[i][d] = cnn_output_buffer[i + 1][d];
		}
	}

	// for each filter == output depth
	CNN_OUTPUT_DEPTH_LEVEL: for (int depth = 0; depth < CNN_KERNEL_COUNT; depth++) {
#pragma HLS pipeline
		// for each data point
		CNN_OUT_DTYPE Ol = 0;
		CNN_LENGTH_LEVEL: for (int l = 0; l < CNN_KERNEL_LENGTH; l++) {
#pragma HLS pipeline
			// for each value in the depth
			
			CNN_OUT_DTYPE Od = 0;
			CNN_KERNEL_DEPTH_LEVEL: for (int d = 0; d < CNN_KERNEL_DEPTH; d++) {
#pragma HLS pipeline
//				Od += input_buffer[l][d] * CNN_weights[CNN_KERNEL_LENGTH - l - 1][d][depth]; // WHY DID KERAS REVERSE THE WEIGHTS?
				Od += input_buffer[l][d] * CNN_weights[l][d][depth];
			}
			Ol += Od;
		}
		cnn_output_buffer[CNN_OUTPUT_LENGTH - 1][depth] = relu<CNN_OUT_DTYPE>(Ol + CNN_bias[depth]);
	}

	// print out the result ofter the convolution
	
	// printf("\nconvoluiton result for the window: \n");
	// for (int d = 0; d < CNN_OUTPUT_DEPTH; d++) {
	// 	printf("%f, ", cnn_output_buffer[CNN_OUTPUT_LENGTH - 1][d]);
	// }
	// printf("\n");
}
