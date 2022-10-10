#ifndef CNN_MODEL
#define CNN_MODEL

#include "cnn.h"
#include "dense_layer.h"

typedef int FUNCTION_SELECT_BIT_WIDTH;

void cnn_action_detection(
		FUNCTION_SELECT_BIT_WIDTH function_select,
		float data_in[CNN_KERNEL_LENGTH*INPUT_DEPTH],
		int &result_out,
		int &data_required,
		float raw_output[DENSE_OUTPUT_NODES],
		float cnn_output[CNN_OUTPUT_LENGTH*CNN_OUTPUT_DEPTH],
		float weights_and_bias[CNN_KERNEL_COUNT * CNN_KERNEL_LENGTH * CNN_KERNEL_DEPTH]);

#endif
