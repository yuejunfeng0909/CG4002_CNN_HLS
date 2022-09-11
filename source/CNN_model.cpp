#include "FIR_CNN.h"
#include "DenseLayer.h"
#include "set_weight_bias.h"
#include <ap_fixed.h>

typedef uint16_t DATA_IN_TYPE;

void top_function(
		ap_uint<3> function_select,
		CNN_RAW_IN_DTYPE data_in[],
		float result_out[],
		INPUT_DTYPE *weights,
		INPUT_DTYPE *bias) {
#pragma HLS interface s_axilite port=return bundle=control

	if (function_select == 0) {
		// set CNN layer weights and bias
		set_CNN_weights_and_bias(weights, bias);
	} else if (function_select == 1) {
		// set Dense layer weights and bias
		set_dense_weights_and_bias(weights, bias);
	} else if (function_select == 2) {
		// input more data
		read_input(data_in);
		compute_convolution();
		compute_dense();
	} else if (function_select == 3) {
		// read result
		copy(dense_output, result_out, DENSE_OUTPUT_NODES);
	} else if (function_select == 4) {
		// reset CNN output buffer
		reset();
	}
}
