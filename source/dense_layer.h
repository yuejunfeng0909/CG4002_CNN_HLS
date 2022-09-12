/*
 * Dense Layers
 * Created: 4 Sep 2022
 * Author: Yue Junfeng
 */

#ifndef DENSE_LAYER
#define DENSE_LAYER

#include "set_weight_bias.h"
#include "FIR_CNN.h"
#include "activation.h"

typedef float DENSE_OUTPUT_DTYPE;

extern DENSE_OUTPUT_DTYPE dense_output[DENSE_OUTPUT_NODES];

void compute_dense();


#endif
