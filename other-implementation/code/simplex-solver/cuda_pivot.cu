//--------------------------------------------------------------------------
//  Cuda Pivot

__global__ void cuda_pivot(int pivot_row, int pivot_col,
		int num_rows, int num_cols,
		float** tableau)
{
	// Keep the pivot value in a register.
	float pivot_val = tableau[pivot_row][pivot_col];

	int row = blockIdx.x;
	int col = threadIdx.x;

	// Calculate new value in tableau
	if (row != pivot_row) {
		float scale = tableau[row][pivot_col]/pivot_val;
		tableau[row][col] -= scale*tableau[pivot_row][col];
	}

}

__global__ void cuda_test(int* vals) {
	*vals = 2;
}