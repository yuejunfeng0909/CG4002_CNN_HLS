/*
 * Dense Layers
 * Created: 4 Sep 2022
 * Author: Yue Junfeng
 */

#ifndef DENSE_LAYER
#define DENSE_LAYER

#include "set_weight_bias.h"
#include "fir_cnn.h"
#include "activation.h"

typedef float DENSE_OUTPUT_DTYPE;

void compute_dense(
		CNN_OUT_DTYPE cnn_output_buffer[CNN_OUTPUT_LENGTH][CNN_OUTPUT_DEPTH],
		DENSE_WEIGHTS_DTYPE dense_weights[DENSE_INPUT_NODES][DENSE_OUTPUT_NODES],
		DENSE_BIAS_DTYPE dense_bias[DENSE_OUTPUT_NODES],
		DENSE_OUTPUT_DTYPE dense_output[DENSE_OUTPUT_NODES]
		);

template<typename T>
void argmax(T input[], int *result){
	DENSE_OUTPUT_DTYPE max = input[0];
	int max_index = 0;
	for (int i = 1; i < DENSE_OUTPUT_NODES; i++) {
		if (input[i] > max) {
			max = input[i];
			max_index = i;
		}
	}
	*result = max_index;
}

#endif
