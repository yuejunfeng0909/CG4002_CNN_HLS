/*
 * Dense Layers
 * Created: 4 Sep 2022
 * Author: Yue Junfeng
 */

#include "DenseLayer.h"

void compute_dense(){
	DENSE_OUTPUT_DTYPE pre_softmax_buffer[DENSE_LAYER_OUTPUT_SIZE];
	CNN_OUT_DTYPE *cnn_output_buffer_alias = &cnn_output_buffer[0][0];

	for (int i = 0; i < DENSE_LAYER_OUTPUT_SIZE; i++) {
#pragma HLS PIPELINE

		// for each output, calculate confidence
		DENSE_OUTPUT_DTYPE output = 0;

		for (int j = 0; j < DENSE_LAYER_INPUT_SIZE; j++) {
			output += cnn_output_buffer_alias[j] * dense_weights[i][j];
		}

		pre_softmax_buffer[i] = output + dense_bias[i];
	}

	// compute softmax and return result
	softmax<DENSE_OUTPUT_DTYPE, float>(pre_softmax_buffer, dense_output, DENSE_LAYER_OUTPUT_SIZE);
}
