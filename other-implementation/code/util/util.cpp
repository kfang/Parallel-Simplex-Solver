#include "util.h"

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
