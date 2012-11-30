//--------------------------------------------------------------------------
//  Cuda Pivot

__global__ void cuda_pivot(int pivot_row, int pivot_col,
		int num_rows, int num_cols,
		float* tableau)
{
	// Keep the pivot value in a register.
	float pivot_val = tableau[pivot_row*num_cols + pivot_col];

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	// Calculate new value in tableau
	if (row != pivot_row && col != pivot_col) {
		float scale = tableau[row*num_cols + pivot_col]/pivot_val;
		tableau[row*num_cols + col] -= scale*tableau[pivot_row*num_cols + col];
		tableau[row*num_cols + col] += 200;
	}

}

__global__ void fix_pivot_col(int pivot_row, int pivot_col, int num_rows, int num_cols, float* tableau) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < num_rows) {
		if (row == pivot_row) {
			tableau[row*num_cols + pivot_col] = 1;
		} else {
			tableau[row*num_cols + pivot_col] = 0;
		}
	}
}

__global__ void cuda2_pivot(int num_cols, float scale, float* cuda_row, float* cuda_pivot_row) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < num_cols) {
		cuda_row[col] -= scale*cuda_pivot_row[col];
	}
}
