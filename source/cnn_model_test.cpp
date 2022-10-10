#include <stdio.h>
#include <vector>
#include <iostream>
#include "cnn_model.h"

#include "cnn_model_test_dataset.h"

int confusion[DENSE_OUTPUT_NODES][DENSE_OUTPUT_NODES];
int accurate_count = 0;

int result;
int result_ready;
float raw_outputs[DENSE_OUTPUT_NODES];
float cnn_average_outputs[CNN_OUTPUT_DEPTH];
float cnn_outputs[CNN_OUTPUT_LENGTH*CNN_OUTPUT_DEPTH];
float weights_and_bias[CNN_KERNEL_COUNT * CNN_KERNEL_LENGTH * CNN_KERNEL_DEPTH];

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
			cnn_action_detection(0, input, result, result_ready, raw_outputs, cnn_average_outputs, cnn_outputs, weights_and_bias);

			// // print cnn outputs
			// printf("CNN outputs: \n");
			// for (int i = 0; i < CNN_OUTPUT_LENGTH; i++) {
			// 	for (int j = 0; j < CNN_OUTPUT_DEPTH; j++) {
			// 		printf("%f ", cnn_outputs[i*CNN_OUTPUT_DEPTH + j]);
			// 	}
			// 	printf("\n");
			// }
			// printf("\n");

			// // print cnn averaged outputs
			// printf("CNN averaged outputs: \n");
			// for (int i = 0; i < CNN_OUTPUT_DEPTH; i++) {
			// 	printf("%f ", cnn_average_outputs[i]);
			// }
			// printf("\n");

			// // print raw outputs
			// printf("current raw outputs: \n");
			// for (int i = 0; i < DENSE_OUTPUT_NODES; i++) {
			// 	printf("%f ", raw_outputs[i]);
			// }
			// printf("\n");

			// printf("\n");
		}
		// print raw outputs
		printf("current raw outputs: \n");
		for (int i = 0; i < DENSE_OUTPUT_NODES; i++) {
			printf("%f ", raw_outputs[i]);
		}
		printf("\n");

		int GOLD_result;
		argmax(test_y[data_index], GOLD_result);

		// add to confusion matrix
		confusion[result][GOLD_result]++;

		if (result == GOLD_result) {
			accurate_count += 1;
		}

		printf("Data %d: predicted = %d vs GOLD = %d\n", data_index, result, GOLD_result);

		// reset
		cnn_action_detection(2, NULL, result, result_ready, raw_outputs, cnn_average_outputs, cnn_outputs, weights_and_bias);
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

void cnn_layer_test() {
    CNN_DTYPE sample_cnn_output_buffer[CNN_OUTPUT_LENGTH][CNN_OUTPUT_DEPTH];
    CNN_DTYPE sample_averaged[CNN_OUTPUT_DEPTH];

    for (int i = 0; i < CNN_OUTPUT_LENGTH; i++) {
        sample_cnn_output_buffer[CNN_OUTPUT_LENGTH-1][0] = 73.0;
        compute_global_average_pool(sample_cnn_output_buffer, sample_averaged);
        printf("sample_averaged[0] = %f\n", sample_averaged[0]);
    }
}

int main() {
	// cnn_layer_test();
	motionDetect();
}
