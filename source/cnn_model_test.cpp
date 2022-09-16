#include <stdio.h>
#include <vector>
#include <iostream>
#include "cnn_model.h"

#include "cnn_model_test_dataset.h"

#define INPUT_DATA_SIZE 12

int confusion[DENSE_OUTPUT_NODES][DENSE_OUTPUT_NODES];
int accurate_count = 0;

int result;
int result_ready;
int debug;

void motionDetect() {
	for (int data_index = 0; data_index < DATASET_SIZE; data_index++) {
		// for each window
		for (int window_start_index = 0; window_start_index < INPUT_LENGTH - CNN_KERNEL_LENGTH + 1; window_start_index += CNN_KERNEL_STRIDE) {
			
			CNN_RAW_IN_DTYPE input[CNN_KERNEL_LENGTH*INPUT_DEPTH];
			// for each frame
			for (int frame = 0; frame < CNN_KERNEL_LENGTH; frame++) {

				// for each channel
				for (int channel = 0; channel < INPUT_DEPTH; channel++) {
					input[frame*INPUT_DEPTH + channel] = CNN_RAW_IN_DTYPE(test_x[data_index][window_start_index + frame][channel]);
				}
			}

			// CNN_RAW_IN_DTYPE *input_ptr = &input[0];
			cnn_action_detection(0, input, result, result_ready, debug);
			printf("result: %d, result_ready: %d, debug: %d\n", result, result_ready, debug);
		}
		cnn_action_detection(1, NULL, result, result_ready, debug);
		printf("result: %d, result_ready: %d, debug: %d\n", result, result_ready, debug);
		int GOLD_result;
		argmax(test_y[data_index], GOLD_result);

		// add to confusion matrix
		confusion[result][GOLD_result]++;

		if (result == GOLD_result) {
			accurate_count += 1;
		}

		printf("Data %d: predicted = %d vs GOLD = %d\n", data_index, result, GOLD_result);

		// reset
		cnn_action_detection(2, NULL, result, result_ready, debug);
	}

	// print accuracy
	printf("Accuracy: %f\n", float(accurate_count) / DATASET_SIZE);

	// print confusion matrix
	for (int i = 0; i < DENSE_OUTPUT_NODES; i++) {
		for (int j = 0; j < DENSE_OUTPUT_NODES; j++) {
			printf("%d   ", confusion[i][j]);
		}
		printf("\n");
	}
}

int main() {
	motionDetect();
}
