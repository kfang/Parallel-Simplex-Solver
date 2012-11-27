////////////////////////////////////////////////////////////////////////////
//                                 util.h                                 //
////////////////////////////////////////////////////////////////////////////

#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <cstdio>
#include <cassert>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

//--------------------------------------------------------------------------
// CREATE_2D_ARRAY

template<class T>
T** create_2D_array(int num_rows, int num_cols)
{
	T** array = new T*[num_rows]();
	assert(array != 0);
	for (int row = 0; row < num_rows; row++) {
		array[row] = new T[num_cols]();
		assert(array[row] != 0);
	}

	return array;
}

//--------------------------------------------------------------------------
// PRINT_MATRIX

inline void print_matrix(int num_rows, int num_cols, float** M)
{
	for (int row = 0; row < num_rows; row++) {
		for (int col = 0; col < num_cols; col++) {
			printf("%8.2f", M[row][col]);
		}
		std::cout << std::endl;
	}
}

inline void print_flat_matrix(int num_rows, int num_cols, float* M)
{
	for (int row = 0; row < num_rows; row++) {
		for (int col = 0; col < num_cols; col++) {
			printf("%8.2f", M[row*num_cols + col]);
		}
		std::cout << std::endl;
	}
}

//--------------------------------------------------------------------------
// TIMESTAMP

double timestamp(void)
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec + 1e-6*tv.tv_usec;
}

#endif
