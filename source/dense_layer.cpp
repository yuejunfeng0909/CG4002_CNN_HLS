/*
 * Dense Layers
 * Created: 4 Sep 2022
 * Author: Yue Junfeng
 */
#include "dense_layer.h"

void compute_dense(
		CNN_OUT_DTYPE cnn_output_buffer[CNN_OUTPUT_LENGTH][CNN_OUTPUT_DEPTH],
		DENSE_WEIGHTS_DTYPE dense_weights[DENSE_OUTPUT_NODES][DENSE_INPUT_NODES],
		DENSE_BIAS_DTYPE dense_bias[DENSE_OUTPUT_NODES],
		DENSE_OUTPUT_DTYPE dense_output[DENSE_OUTPUT_NODES]){
#pragma HLS ARRAY_PARTITION variable=cnn_output_buffer type=cyclic dim=2 complete
#pragma HLS ARRAY_PARTITION variable=dense_weights type=cyclic dim=2 complete
#pragma HLS ARRAY_PARTITION variable=dense_bias type=complete
	// DENSE_OUTPUT_DTYPE pre_softmax_buffer[DENSE_OUTPUT_NODES];
	CNN_OUT_DTYPE *cnn_output_buffer_alias = &cnn_output_buffer[0][0];

	for (int i = 0; i < DENSE_OUTPUT_NODES; i++) {
#pragma HLS pipeline

		// for each output, calculate confidence
		DENSE_OUTPUT_DTYPE output = 0;

		for (int j = 0; j < DENSE_INPUT_NODES; j++) {
#pragma HLS unroll factor = 11
			output += cnn_output_buffer_alias[j] * dense_weights[i][j];
		}

		dense_output[i] = output + dense_bias[i];
	}

	// compute softmax and return result
//	softmax<DENSE_OUTPUT_DTYPE, DENSE_OUTPUT_DTYPE>(pre_softmax_buffer, dense_output, DENSE_OUTPUT_NODES);
//	copy<DENSE_OUTPUT_DTYPE, DENSE_OUTPUT_DTYPE>(pre_softmax_buffer, dense_output, DENSE_OUTPUT_NODES);
}
