#ifndef CNN_MODEL
#define CNN_MODEL

#include "cnn.h"
#include "dense_layer.h"

typedef int FUNCTION_SELECT_BIT_WIDTH;

void cnn_action_detection(
		FUNCTION_SELECT_BIT_WIDTH function_select,
		float data_0,
		float data_1,
		float data_2,
		float data_3,
		float data_4,
		float data_5,
		float raw_output[DENSE_OUTPUT_NODES],
		float weights_and_bias[CNN_KERNEL_COUNT * CNN_KERNEL_LENGTH * CNN_KERNEL_DEPTH]);

#endif
