/*
 * Rectified Linear Unit and softmax
 * Created: 4 Sep 2022
 * Author: Yue Junfeng
 */

#ifndef ACTIVATION
#define ACTIVATION

//#include "hls_math.h"

template <typename DTYPE>
DTYPE relu(DTYPE x){
#pragma HLS INLINE
	return x > 0 ? x : 0;
}

//template <typename INTYPE, typename OUTTYPE>
//void softmax(INTYPE x[], OUTTYPE y[], int size);

#endif
