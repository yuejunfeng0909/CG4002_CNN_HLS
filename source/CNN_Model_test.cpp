#include <stdio.h>
#include "CNN_Model.h"

int main() {

	CNN_IN_DTYPE matrix_conv_input[INPUT_DEPTH][INPUT_LENGTH] = {
			1.1f, 1.2f, 1.3f, 0.1f, 0.9f,
			2.1f, 1.2f, 2.3f, 0.9f, 0.1f,
			1.1f, 2.2f, 1.3f, 1.0f, 0.1f};

	CNN_WEIGHT_DTYPE matrix_conv_weights[CNN_KERNEL_DEPTH][CNN_KERNEL_LENGTH] = {
			0.1f, 0.2f, 0.3f,
			0.4f, 0.5f, 0.6f,
			0.7f, 0.8f, 0.9f};

	CNN_OUT_DTYPE matrix_conv_output[OUTPUT_DEPTH][OUTPUT_LENGTH];

	CNN_OUT_DTYPE matrix_conv_output_GOLD[OUTPUT_DEPTH][OUTPUT_LENGTH] = {
			6.5400f, 5.4400f, 4.3500f};


	matrixConv(matrix_conv_input, matrix_conv_weights, matrix_conv_output);

	int retval = 0;
	// Check outputs against expected
	for (int i = 0; i < OUTPUT_DEPTH; i++) {
		for (int j = 0; j < OUTPUT_LENGTH; j++) {
			if(matrix_conv_output[i][j] - matrix_conv_output_GOLD[i][j] > 0.0001f){
				printf("mismatch at depth=%d, length=%d\n", i, j);
				printf("output vs golden = %.3f vs %.3f\n", matrix_conv_output[i][j], matrix_conv_output_GOLD[i][j]);
				retval = 1;
			}
		}
	}

	// Print Results
	if(retval == 0){
		printf("    *** *** *** *** \n");
		printf("    Results are good \n");
		printf("    *** *** *** *** \n");
	} else {
		printf("    *** *** *** *** \n");
		printf("    Mismatch: retval=%d \n", retval);
		printf("    *** *** *** *** \n");
	}

	return retval;
}

