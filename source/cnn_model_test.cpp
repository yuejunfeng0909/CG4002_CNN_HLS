#include <stdio.h>
#include <vector>
#include <iostream>
#include "cnn_model.h"

#include "cnn_model_test_dataset.h"

#define INPUT_DATA_SIZE 12

int confusion[DENSE_OUTPUT_NODES][DENSE_OUTPUT_NODES];
int accurate_count = 0;

void motionDetect() {

	int func_select = 0;
	// for each data
	for (int data_index = 0; data_index < DATASET_SIZE; data_index++) {
		// for each window
		for (int window_start_index = 0; window_start_index < INPUT_LENGTH - CNN_KERNEL_LENGTH + 1; window_start_index += CNN_KERNEL_STRIDE) {
			// printf("\nNew window from %d to %d\n", window_start_index, window_start_index + CNN_KERNEL_LENGTH);
			
			CNN_RAW_IN_DTYPE input[CNN_KERNEL_LENGTH][INPUT_DEPTH];
			// for each frame
			for (int frame = 0; frame < CNN_KERNEL_LENGTH; frame++) {

				// for each channel
				for (int channel = 0; channel < INPUT_DEPTH; channel++) {
					input[frame][channel] = CNN_RAW_IN_DTYPE(test_x[data_index][window_start_index + frame][channel]);
				}
			}

//			 print out the window
//			 for (int frame = 0; frame < CNN_KERNEL_LENGTH; frame++) {
//			 	for (int channel = 0; channel < INPUT_DEPTH; channel++) {
//			 		printf("%d, ", input[frame][channel]);
//			 	}
//			 	printf("\n");
//			 }
//			 printf("\n");
			CNN_RAW_IN_DTYPE *input_ptr = &input[0][0];
			func_select = 0;
			cnn_action_detection(&func_select, input_ptr, NULL);
		}
		func_select = 1;
		int *result_ptr;
		cnn_action_detection(&func_select, NULL, result_ptr);
		int GOLD_result;
		argmax(test_y[data_index], &GOLD_result);

		// add to confusion matrix
		confusion[*result_ptr][GOLD_result]++;

		if (*result_ptr == GOLD_result) {
			accurate_count += 1;
		}

		printf("Data %d: predicted = %d vs GOLD = %d\n", data_index, *result_ptr, GOLD_result);
		
		func_select = 2;
		// reset
		cnn_action_detection(&func_select, NULL, NULL);
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
