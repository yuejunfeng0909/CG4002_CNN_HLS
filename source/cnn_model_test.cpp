#include <stdio.h>
#include <vector>
#include <iostream>
#include "cnn_model.h"

#include "cnn_model_test_dataset.h"

#define INPUT_DATA_SIZE 12

void motionDetect() {
	// for each data
	for (int data_index = 0; data_index < DATASET_SIZE; data_index++) {
		// for each frame
		for (int frame = 0; frame < INPUT_LENGTH; frame++) {
			CNN_RAW_IN_DTYPE input[INPUT_DEPTH];

			// for each channel
			for (int channel = 0; channel < INPUT_DEPTH; channel++) {
				input[channel] = CNN_RAW_IN_DTYPE(test_x[data_index][frame][channel]);
			}
			top_function(0, input, NULL);
		}
		float results[DENSE_OUTPUT_NODES];
		top_function(1, NULL, results);
		printf("\nData %d:\n", data_index);
		for (int i = 0; i < DENSE_OUTPUT_NODES; i++) {
			printf("for class %d, predicted = %f vs GOLD = %d\n", i, results[i], test_y[data_index][i]);
		}

		// reset
		top_function(2, NULL, NULL);
	}


}

int main() {
	motionDetect();
}
