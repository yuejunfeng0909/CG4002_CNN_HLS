/*
 * Dense Layers
 * Created: 4 Sep 2022
 * Author: Yue Junfeng
 */

#ifndef DENSE_LAYER
#define DENSE_LAYER

#include "set_weight_bias.h"
#include "FIR_CNN.h"

typedef float DENSE_OUTPUT_DTYPE;

#define DENSE_LAYER_INPUT_SIZE 736
#define DENSE_LAYER_OUTPUT_SIZE 3

DENSE_OUTPUT_DTYPE dense_output[DENSE_LAYER_OUTPUT_SIZE];

void compute_dense();


#endif
