/*
 * Dense Layers
 * Created: 4 Sep 2022
 * Author: Yue Junfeng
 */

#ifndef DENSE_LAYER
#define DENSE_LAYER

#include "fir_cnn.h"

#define DENSE_INPUT_NODES CNN_OUTPUT_LENGTH * CNN_KERNEL_COUNT
#define DENSE_OUTPUT_NODES 3

typedef float DENSE_DTYPE;

void compute_dense(
		CNN_DTYPE cnn_output_buffer[CNN_OUTPUT_LENGTH][CNN_OUTPUT_DEPTH],
		DENSE_DTYPE dense_weights[DENSE_INPUT_NODES][DENSE_OUTPUT_NODES],
		DENSE_DTYPE dense_bias[DENSE_OUTPUT_NODES],
		DENSE_DTYPE dense_output[DENSE_OUTPUT_NODES]
		);

template<typename T>
void argmax(T input[], int &result){
	DENSE_DTYPE max = input[0];
	int max_index = 0;
	ARGMAX: for (int i = 1; i < DENSE_OUTPUT_NODES; i++) {
		if (input[i] > max) {
			max = input[i];
			max_index = i;
		}
	}
	result = max_index;
}

#endif
