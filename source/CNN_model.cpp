#include "cnn_model.h"

void top_function(
		FUNCTION_SELECT_BIT_WIDTH function_select,
		CNN_RAW_IN_DTYPE data_in[DENSE_OUTPUT_NODES],
		float result_out[DENSE_OUTPUT_NODES]) {
#pragma HLS interface s_axilite port=return bundle=control

	if (function_select == 0) {
		// input more data
		read_input(data_in);
		compute_convolution();
		compute_dense();
	} else if (function_select == 1) {
		// read result
		copy(dense_output, result_out, DENSE_OUTPUT_NODES);
	} else if (function_select == 2) {
		// reset CNN output buffer
		reset();
	}
}
