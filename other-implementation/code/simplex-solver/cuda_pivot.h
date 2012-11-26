/*
 * cuda_pivot.h
 *
 *  Created on: Nov 25, 2012
 *      Author: akaiser
 */

#ifndef CUDA_PIVOT_H_
#define CUDA_PIVOT_H_

__global__ void cuda_pivot(int pivot_row, int pivot_col,
		int num_rows, int num_cols,
		float* tableau);

__global__ void cuda_test(int* vals);

#endif /* CUDA_PIVOT_H_ */
