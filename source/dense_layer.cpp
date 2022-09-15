/*
 * Dense Layers
 * Created: 4 Sep 2022
 * Author: Yue Junfeng
 */
#include "dense_layer.h"
// #include <stdio.h>

void compute_dense(
		CNN_OUT_DTYPE cnn_output_buffer[CNN_OUTPUT_LENGTH][CNN_OUTPUT_DEPTH],
		DENSE_WEIGHTS_DTYPE dense_weights[DENSE_INPUT_NODES][DENSE_OUTPUT_NODES],
		DENSE_BIAS_DTYPE dense_bias[DENSE_OUTPUT_NODES],
		DENSE_OUTPUT_DTYPE dense_output[DENSE_OUTPUT_NODES]
		){
#pragma HLS ARRAY_PARTITION variable=dense_weights dim=1 complete
#pragma HLS ARRAY_PARTITION variable=dense_bias type=complete
#pragma HLS INLINE

	for (int i = 0; i < DENSE_OUTPUT_NODES; i++) {
#pragma HLS pipeline

		// for each output, calculate confidence
		DENSE_OUTPUT_DTYPE output = 0;

		for (int j = 0; j < DENSE_INPUT_NODES; j++) {
			output += cnn_output_buffer[j/CNN_OUTPUT_DEPTH][j%CNN_OUTPUT_DEPTH] * dense_weights[j][i];
		}

		dense_output[i] = output + dense_bias[i];
	}
}
