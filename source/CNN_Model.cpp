/*
 * Create a matrix convolution IP first
 */

#include "CNN_Model.h"

CNN_IN_DTYPE input_buffer[INPUT_DEPTH][INPUT_LENGTH];

CNN_WEIGHT_DTYPE weights_buffer[CNN_KERNEL_DEPTH][CNN_KERNEL_LENGTH] = {
		0.1f, 0.2f, 0.3f,
		0.4f, 0.5f, 0.6f,
		0.7f, 0.8f, 0.9f};

CNN_OUT_DTYPE output_buffer[OUTPUT_DEPTH][OUTPUT_LENGTH];

void matrixConv(
		CNN_IN_DTYPE input[INPUT_DEPTH][INPUT_LENGTH],
		CNN_WEIGHT_DTYPE weights[CNN_KERNEL_DEPTH][CNN_KERNEL_LENGTH],
		CNN_OUT_DTYPE output[OUTPUT_DEPTH][OUTPUT_LENGTH]){

	// iterate for each element in the output
	for (int OD = 0; OD < OUTPUT_DEPTH; OD++) {							// Output depth
		for (int OL = 0; OL < OUTPUT_LENGTH; OL++) {					// Output length

			// iterate to accumulate the value
			CNN_OUT_DTYPE accumulate = 0.0f;

			for (int KD = 0; KD < CNN_KERNEL_DEPTH; KD++) {				// Kernel depth
				for (int KL = 0; KL < CNN_KERNEL_LENGTH; KL++) {		// Kernel length

					accumulate +=
							weights[CNN_KERNEL_DEPTH - KD - 1][CNN_KERNEL_LENGTH - KL - 1] *
							input[OD * CNN_KERNEL_STRIDE + KD][OL * CNN_KERNEL_STRIDE + KL];
				}
			}

			output[OD][OL] = accumulate;

		}
	}

}
