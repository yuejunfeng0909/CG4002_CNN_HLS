/*
 * Dense Layers
 * Created: 4 Sep 2022
 * Author: Yue Junfeng
 */
#include "dense_layer.h"

DENSE_OUTPUT_DTYPE dense_output[DENSE_OUTPUT_NODES];

void compute_dense(){
	// DENSE_OUTPUT_DTYPE pre_softmax_buffer[DENSE_OUTPUT_NODES];
	CNN_OUT_DTYPE *cnn_output_buffer_alias = &cnn_output_buffer[0][0];

	for (int i = 0; i < DENSE_OUTPUT_NODES; i++) {
#pragma HLS PIPELINE

		// for each output, calculate confidence
		DENSE_OUTPUT_DTYPE output = 0;

		for (int j = 0; j < DENSE_INPUT_NODES; j++) {
			output += cnn_output_buffer_alias[j] * dense_weights[i][j];
		}

		dense_output[i] = output + dense_bias[i];
	}

	// compute softmax and return result
//	softmax<DENSE_OUTPUT_DTYPE, DENSE_OUTPUT_DTYPE>(pre_softmax_buffer, dense_output, DENSE_OUTPUT_NODES);
//	copy<DENSE_OUTPUT_DTYPE, DENSE_OUTPUT_DTYPE>(pre_softmax_buffer, dense_output, DENSE_OUTPUT_NODES);
}
