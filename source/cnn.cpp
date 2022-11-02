#include "cnn.h"

float relu(float x){
	return x > 0 ? x : 0;
}

void compute_convolution(
		CNN_DTYPE input_buffer[CNN_KERNEL_LENGTH][INPUT_DEPTH],
		CNN_DTYPE CNN_weights[CNN_KERNEL_LENGTH][CNN_KERNEL_DEPTH][CNN_KERNEL_COUNT],
		CNN_DTYPE CNN_bias[CNN_KERNEL_COUNT],
		CNN_DTYPE cnn_output_buffer[CNN_OUTPUT_LENGTH][CNN_OUTPUT_DEPTH]) {
	// shift output register
	CNN_REGISTER_SHIFT: 
	for (int i = 0; i < CNN_OUTPUT_LENGTH - 1; i++) {
		for (int d = 0; d < CNN_OUTPUT_DEPTH; d++) {
			cnn_output_buffer[i][d] = cnn_output_buffer[i + 1][d];
		}
	}

	// for each filter == output depth
	CNN_OUTPUT_DEPTH_LEVEL: for (int depth = 0; depth < CNN_KERNEL_COUNT; depth++) {
#pragma HLS PIPELINE II=10
		// for each data point
		CNN_DTYPE Ol = 0;
		CNN_LENGTH_LEVEL: for (int l = 0; l < CNN_KERNEL_LENGTH; l++) {
			// for each value in the depth
			
			CNN_DTYPE Od = 0;
			CNN_KERNEL_DEPTH_LEVEL: for (int d = 0; d < CNN_KERNEL_DEPTH; d++) {
//				Od += input_buffer[l][d] * CNN_weights[CNN_KERNEL_LENGTH - l - 1][d][depth]; // WHY DID KERAS REVERSE THE WEIGHTS?
				Od += input_buffer[l][d] * CNN_weights[l][d][depth];
			}
			Ol += Od;
		}
		cnn_output_buffer[CNN_OUTPUT_LENGTH - 1][depth] = relu(Ol + CNN_bias[depth]);
	}
}

void compute_global_average_pool(
		CNN_DTYPE cnn_output_buffer[CNN_OUTPUT_LENGTH][CNN_OUTPUT_DEPTH],
		CNN_DTYPE averaged[CNN_OUTPUT_DEPTH]) {
	
// 	for (int d = 0; d < CNN_OUTPUT_DEPTH; d++) {
// #pragma HLS PIPELINE II=5
// 		CNN_DTYPE sum = 0;
// 		for (int i = 0; i < CNN_OUTPUT_LENGTH; i++) {
// 			sum += cnn_output_buffer[i][d];
// 		}
// 		averaged[d] = sum / CNN_OUTPUT_LENGTH;
// 	}
	for (int d = 0; d < CNN_OUTPUT_DEPTH; d++) {
		averaged[d] += cnn_output_buffer[CNN_OUTPUT_LENGTH - 1][d]/CNN_OUTPUT_LENGTH;
		averaged[d] -= cnn_output_buffer[0][d]/CNN_OUTPUT_LENGTH;
	}
}

// void compute_global_average_pool(
// 		CNN_DTYPE cnn_output_buffer[CNN_OUTPUT_LENGTH][CNN_OUTPUT_DEPTH],
// 		CNN_DTYPE averaged[CNN_OUTPUT_DEPTH]) {
	
// 	static CNN_DTYPE sum[CNN_OUTPUT_DEPTH] = {0};
// 	static CNN_DTYPE next_discard[CNN_OUTPUT_DEPTH] = {0};

// 	// sum subtract by discard
// 	for (int d = 0; d < CNN_OUTPUT_DEPTH; d++) {
// 		sum[d] -= next_discard[d];
// 	}
// 	for (int d = 0; d < CNN_OUTPUT_DEPTH; d++) {
// 		sum[d] += cnn_output_buffer[CNN_OUTPUT_LENGTH-1][d];
// 	}
// 	for (int d = 0; d < CNN_OUTPUT_DEPTH; d++) {
// 		next_discard[d] = cnn_output_buffer[0][d];
// 	}
	
// 	// compute average
// 	for (int d = 0; d < CNN_OUTPUT_DEPTH; d++) {
// 		averaged[d] = sum[d] / CNN_OUTPUT_LENGTH;
// 	}
// }