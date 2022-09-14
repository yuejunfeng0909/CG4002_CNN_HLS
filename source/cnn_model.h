#ifndef CNN_MODEL
#define CNN_MODEL

#include "fir_cnn.h"
#include "dense_layer.h"
#include "set_weight_bias.h"

typedef int FUNCTION_SELECT_BIT_WIDTH;

extern DENSE_OUTPUT_DTYPE dense_output[DENSE_OUTPUT_NODES];

void cnn_action_detection(
		FUNCTION_SELECT_BIT_WIDTH *function_select,
		CNN_RAW_IN_DTYPE *data_in,
		int *result_out);

#endif
