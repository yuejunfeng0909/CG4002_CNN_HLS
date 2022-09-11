/*
 * Rectified Linear Unit
 * Created: 4 Sep 2022
 * Author: Yue Junfeng
 */

#ifndef ACTIVATION
#define ACTIVATION

#include "hls_math.h"

template <typename DTYPE>
void relu(DTYPE x, DTYPE *y);

template <typename INTYPE, typename OUTTYPE>
void softmax(INTYPE *x, OUTTYPE *y, int size);

#endif
