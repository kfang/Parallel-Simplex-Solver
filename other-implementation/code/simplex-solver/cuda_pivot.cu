//--------------------------------------------------------------------------
//  Cuda Pivot

__global__ void cuda_pivot(int* pivot_row_loc, int* pivot_col_loc,
		int num_rows, int num_cols,
		float* pivot_val, float* tableau)
{
	int pivot_row = *pivot_row_loc;
	int pivot_col = *pivot_col_loc;
	// Keep the pivot value in a register.
	float pivot_val = tableau[pivot_row*num_cols + pivot_col];

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	// Calculate new value in tableau
	if (row < num_rows && col < num_cols) {
		if (row != pivot_row && col != pivot_col) {
			float scale = tableau[row*num_cols + pivot_col]/pivot_val;
			tableau[row*num_cols + col] -= scale*tableau[pivot_row*num_cols + col];
		} else if (row == pivot_row && col == pivot_col) {
			*pivot_val = tableau[row*num_cols + col];
		}
	}

}

__global__ void fix_pivot_col(int* pivot_row_loc, int* pivot_col_loc, int num_rows, int num_cols, float* tableau) {
	int pivot_row = *pivot_row_loc;
	int pivot_col = *pivot_col_loc;

	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < num_rows) {
		if (row == pivot_row) {
			tableau[row*num_cols + pivot_col] = 1;
		} else {
			tableau[row*num_cols + pivot_col] = 0;
		}
	}

}

__global__ void scale_pivot_row(int* pivot_row_loc, int* pivot_col_loc, int num_rows, int num_cols, float* pivot_val, float* tableau) {
	int pivot_row = *pivot_row_loc;
	int pivot_col = *pivot_col_loc;
	
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < num_cols) {
		tableau[pivot_row*num_cols + col] = tableau[pivot_row*num_cols + col] / (*pivot_val);
	}
}

__global__ void find_pivot_row_and_col(int* pivot_row, int* pivot_col, float* tableau, int num_rows, int num_cols, bool* done) {

	float min_val = tableau[0];
	*pivot_col = 0;
	for(int i = 0; (i < num_cols-1); i++) {
		if (tableau[i] < min_val) {
			min_val = tableau[i];
			(*pivot_col) = i;
		}
	}
	for((*pivot_row) = 1; ((*pivot_row) < num_rows) && (tableau[(*pivot_row)*num_cols + (*pivot_col)] <= 0); (*pivot_row)++);
	for(int i = (*pivot_row)+1; i < num_rows; i++) {
		if (tableau[i*num_cols + (*pivot_col)] > 0) {
			if (tableau[i*num_cols + (num_cols -1)]/tableau[i*num_cols + (*pivot_col)] < tableau[(*pivot_row)*num_cols + num_cols-1]/tableau[(*pivot_row)*num_cols + (*pivot_col)]) {
				(*pivot_row) = i;
			}
		}
	}
	if (min_val >= 0) {
		*done = true;
	} else {
		*done = false;
	}
	//Deal with unbounded at some point
}

__global__ void cuda2_pivot(int num_cols, float scale, float* cuda_row, float* cuda_pivot_row) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < num_cols) {
		cuda_row[col] -= scale*cuda_pivot_row[col];
	}
}
