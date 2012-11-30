/*
 * cuda_pivot.h
 *
 *  Created on: Nov 25, 2012
 *      Author: akaiser
 */

#ifndef CUDA_PIVOT_H_
#define CUDA_PIVOT_H_

__global__ void cuda_pivot(int* pivot_row_loc, int* pivot_col_loc,
		int num_rows, int num_cols,
		float* pivot_val, float* tableau);

__global__ void fix_pivot_col(int* pivot_row_loc, int* pivot_col_loc, int num_rows, int num_cols, float* tableau);

__global__ void scale_pivot_row(int* pivot_row_loc, int* pivot_col_loc, int num_rows, int num_cols, float* pivot_val, float* tableau);

__global__ void find_pivot_row_and_col(int* pivot_row_loc, int* pivot_col_loc, float* tableau, int num_rows, int num_cols, bool* done);

__global__ void cuda2_pivot(int num_cols, float scale, float* cuda_row, float* cuda_pivot_row);

#endif /* CUDA_PIVOT_H_ */
