/*
 * Dense Layers
 * Created: 4 Sep 2022
 * Author: Yue Junfeng
 */
#include "dense_layer.h"

void compute_dense(
		CNN_DTYPE input_buffer[CNN_OUTPUT_DEPTH],
		DENSE_DTYPE dense_weights[DENSE_INPUT_NODES][DENSE_OUTPUT_NODES],
		DENSE_DTYPE dense_bias[DENSE_OUTPUT_NODES],
		DENSE_DTYPE dense_output[DENSE_OUTPUT_NODES]
		){

	DENSE_OUTPUT: for (int i = 0; i < DENSE_OUTPUT_NODES; i++) {
#pragma HLS PIPELINE II=17

		// for each output, calculate confidence
		DENSE_DTYPE output = 0;

		DENSE_INPUT: for (int j = 0; j < DENSE_INPUT_NODES; j++) {
			output += input_buffer[j] * dense_weights[j][i];
		}

		dense_output[i] = output + dense_bias[i];
	}
}
