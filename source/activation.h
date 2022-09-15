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
	return x > 0 ? x : 0;
}

#endif
