#include "fir_cnn.h"
#include <stdio.h>

void read_input(
		CNN_RAW_IN_DTYPE input[INPUT_DEPTH],
		CNN_IN_DTYPE input_buffer[CNN_KERNEL_LENGTH][INPUT_DEPTH]){
#pragma HLS ARRAY_PARTITION variable=input type=complete
	for (int d = 0; d < INPUT_DEPTH; d++) {
#pragma HLS unroll
		for (int i = CNN_KERNEL_LENGTH - 1; i > 0; i--) {
#pragma HLS unroll
			input_buffer[i][d] = input_buffer[i-1][d];
		}
		input_buffer[0][d] = (CNN_IN_DTYPE)input[d];
	}
}

void reset(CNN_OUT_DTYPE cnn_output_buffer[CNN_OUTPUT_LENGTH][CNN_OUTPUT_DEPTH]) {
#pragma HLS ARRAY_PARTITION variable=cnn_output_buffer type=cyclic dim=1 complete
	for (int i = 0; i < CNN_OUTPUT_LENGTH; i++) {
#pragma HLS unroll
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
#pragma HLS ARRAY_PARTITION variable=input_buffer type=cyclic dim=2 complete
#pragma HLS ARRAY_PARTITION variable=CNN_weights type=cyclic dim=2 complete
#pragma HLS ARRAY_PARTITION variable=CNN_bias type=complete

	// before shift
	//  printf("\n--------------------------------------------------\n");
	//  printf("\noutput_buffer before shift: \n");
	//  for (int i = 0; i < CNN_OUTPUT_LENGTH; i++) {
	//  	for (int d = 0; d < CNN_OUTPUT_DEPTH; d++) {
	//  		printf("%f, ", cnn_output_buffer[i][d]);
	//  	}
	//  	printf("\n");
	//  }

	// shift output register
	for (int d = 0; d < CNN_OUTPUT_DEPTH; d++) {
#pragma HLS pipeline
		for (int i = 0; i < CNN_OUTPUT_LENGTH - 1; i++) {
			cnn_output_buffer[i][d] = cnn_output_buffer[i + 1][d];
		}
	}

	// after shift
	// printf("\noutput_buffer after shift: \n");
	// for (int i = 0; i < CNN_OUTPUT_LENGTH; i++) {
	// 	for (int d = 0; d < CNN_OUTPUT_DEPTH; d++) {
	// 		printf("%f, ", cnn_output_buffer[i][d]);
	// 	}
	// 	printf("\n");
	// }

	// for each filter == output depth
	for (int depth = 0; depth < CNN_KERNEL_COUNT; depth++) {
#pragma HLS pipeline
		// for each data point
		CNN_OUT_DTYPE Oi = 0;
		for (int l = 0; l < CNN_KERNEL_LENGTH; l++) {
#pragma HLS unroll
			// for each value in the depth
			for (int d = 0; d < CNN_KERNEL_DEPTH; d++) {
//				Oi += input_buffer[l][d] * CNN_weights[CNN_KERNEL_LENGTH - l - 1][d][depth]; // WHY DID KERAS REVERSE THE WEIGHTS?
				Oi += input_buffer[l][d] * CNN_weights[l][d][depth];
			}
		}
		cnn_output_buffer[CNN_OUTPUT_LENGTH - 1][depth] = relu<CNN_OUT_DTYPE>(Oi + CNN_bias[depth]);
	}

	// print out the result ofter the convolution
	
	// printf("\nconvoluiton result for the window: \n");
	// for (int d = 0; d < CNN_OUTPUT_DEPTH; d++) {
	// 	printf("%f, ", cnn_output_buffer[CNN_OUTPUT_LENGTH - 1][d]);
	// }
	// printf("\n");
}
