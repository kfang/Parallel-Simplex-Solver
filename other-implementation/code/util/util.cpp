#include "util.h"
#include <iostream>
#include <cstdio>
#include <cassert>

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

void print_matrix(int num_rows, int num_cols, float** M)
{
	for (int row = 0; row < num_rows; row++) {
		for (int col = 0; col < num_cols; col++) {
			printf("%8.2f", M[row][col]);
		}
		std::cout << std::endl;
	}
}
