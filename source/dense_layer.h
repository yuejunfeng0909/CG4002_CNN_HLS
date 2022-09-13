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
		CNN_OUT_DTYPE cnn_output_buffer[OUTPUT_LENGTH][OUTPUT_DEPTH],
		DENSE_WEIGHTS_DTYPE dense_weights[DENSE_OUTPUT_NODES][DENSE_INPUT_NODES],
		DENSE_BIAS_DTYPE dense_bias[DENSE_OUTPUT_NODES],
		DENSE_OUTPUT_DTYPE dense_output[DENSE_OUTPUT_NODES]);


#endif
