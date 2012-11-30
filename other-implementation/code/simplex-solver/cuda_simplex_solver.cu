////////////////////////////////////////////////////////////////////////////
//                       serial_simplex_solver.cpp                        //
////////////////////////////////////////////////////////////////////////////


#include <iostream>
#include <cstdio>
#include <string>
#include "cuda_simplex_solver.h"
#include "simplex_problem.h"
#include "simplex_solution.h"
#include "util.h"
#include "cuda_pivot.h"

//--------------------------------------------------------------------------
// CONSTRUCTORS AND DESTRUCTOR

Cuda_Simplex_Solver::Cuda_Simplex_Solver(void)
{
}

Cuda_Simplex_Solver::~Cuda_Simplex_Solver(void)
{
}

//--------------------------------------------------------------------------
// SOLVE

Simplex_Solution Cuda_Simplex_Solver::solve(Simplex_Problem& problem)
{
	// Make a new tableau for solving the problem.
	float** tableau = create_tableau(problem);

	// Get the number of variables and constraints in the problem.
	int num_variables = problem.get_num_variables();
	int num_constraints = problem.get_num_constraints();

	// Calculate the number of rows and columns in the tableau and allocate memory.
	int num_rows = num_constraints + 1;
	int num_cols = num_variables + num_constraints + 1;

	double time = timestamp();

	float* flat_tableau;
	flat_tableau = (float *) malloc(num_rows*num_cols*sizeof(float));
	for (int i = 0; i < num_rows; i++) {
		for (int j = 0; j < num_cols; j++) {
			flat_tableau[i*num_cols + j] = tableau[i][j];
		}
	}

	std::cerr << "Setup done" << std::endl;

	//Cuda Pointer and mem
	float* cuda_tableau;

	//Make device space
	if (cudaMalloc((void**)&cuda_tableau, num_rows*num_cols*sizeof(float)) != cudaSuccess) {
		std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;
		std::cerr << "First Malloc Failed" << std::endl;
        exit(1);
	}

	// Copy over tableau
	if (cudaMemcpy(cuda_tableau, flat_tableau, num_rows*num_cols*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
		std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;
		std::cerr << "Failed on first tableau copy" << std::endl;
        exit(1);
	}

	// While the objective function can be increased, find a better
	// vertex on the simplex.
	int pivot_col, pivot_row;
	int d_pivot_col, d_pivot_row;
	bool done;
	bool d_done;
	for (;;) {
		find_pivot_row_and_col<<<1,1>>>(d_pivot_row, d_pivot_col, tableau, num_rows, num_cols, d_done);

		if (cudaMemcpy(&done, d_done, sizeof(bool), cudaMemcpyDeviceToHost) != cudaSuccess) {
			std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;
			std::cerr << "Failed to copy back" << std::endl;
	        exit(1);
		}

		if (done) {
			break;
		}

		//std::cerr << "---------------------------------" << std::endl;
		//std::cerr << "BEFORE PIVOT" << std::endl;
		//print_flat_matrix(num_rows, num_cols, flat_tableau);
		//std::cerr << "pivot_row: " << pivot_row << std::endl;
		//std::cerr << "pivot_col: " << pivot_col << std::endl;
		//std::cerr << "AFTER PIVOT" << std::endl;
		pivot(&d_pivot_row, &d_pivot_col_loc, num_rows, num_cols, flat_tableau, cuda_tableau);
		//print_flat_matrix(num_rows, num_cols, flat_tableau);
	}

	float ans;

	// Copy back
	if (cudaMemcpy(&ans, cuda_tableau + num_cols - 1, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
		std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;
		std::cerr << "Failed to copy back" << std::endl;
        exit(1);
	}

	cudaFree(cuda_tableau);

	time = timestamp() - time;
	std::cerr << "Solve time: " << time << std::endl;

	std::cerr << "DONE!!!" << std::endl;
	std::cerr << "Max value: " << ans << std::endl;

	std::cout << num_variables << "," << time << std::endl;

	return Simplex_Solution();
}

//--------------------------------------------------------------------------
// PIVOT

void Cuda_Simplex_Solver::pivot(const int& pivot_row, const int& pivot_col,
                            const int& num_rows, const int& num_cols,
                            float* tableau, float* cuda_tableau)
{


	// Do Pivot
	dim3 threads(16,16);
	int num_blocks = std::max(num_rows, num_cols);
	num_blocks = ceil((num_blocks+15)/16);
	dim3 blocks(num_blocks, num_blocks);

	cuda_pivot <<< blocks, threads >>> (pivot_row, pivot_col, num_rows, num_cols, cuda_tableau);
	
	cudaThreadSynchronize();

	if (cudaGetLastError() != cudaSuccess) {
		std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;
		std::cerr << "Kernel Failed" << std::endl;
        exit(1);
	}
	
	scale_pivot_row <<< (num_cols+127), 128 >>> (pivot_row, pivot_col, num_rows, num_cols, cuda_tableau);

	cudaThreadSynchronize();

	if (cudaGetLastError() != cudaSuccess) {
		std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;
		std::cerr << "Kernel Failed" << std::endl;
        exit(1);
	}

	fix_pivot_col <<< (num_rows+127), 128 >>> (pivot_row, pivot_col, num_rows, num_cols, cuda_tableau);

	cudaThreadSynchronize();

	if (cudaGetLastError() != cudaSuccess) {
		std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;
		std::cerr << "Kernel Failed" << std::endl;
        exit(1);
	}

	// Copy back
	if (cudaMemcpy(tableau, cuda_tableau, num_rows*num_cols*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
		std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;
		std::cerr << "Failed to copy back" << std::endl;
        exit(1);
	}

}


//--------------------------------------------------------------------------
// CREATE_TABLEAU

float** Cuda_Simplex_Solver::create_tableau(Simplex_Problem& problem)
{
	// Get the number of variables and constraints in the problem.
	int num_variables = problem.get_num_variables();
	int num_constraints = problem.get_num_constraints();

	// Calculate the number of rows and columns in the tableau and allocate memory.
	int num_rows = num_constraints + 1;
	int num_cols = num_variables + num_constraints + 1;
	float** tableau = create_2D_array<float>(num_rows, num_cols);

	// Add the objective function to the 0th row of the tableau.
	add_obj_func_to_tableau(num_rows, num_cols, tableau, problem);

	// Add the constraints to the rest of the rows of the tableau.
	add_constraints_to_tableau(num_rows, num_cols, tableau, problem);

	// The tableau is finished!
	return tableau;
}

//--------------------------------------------------------------------------
// ADD_OBJ_FUNC_TO_TABLEAU

void Cuda_Simplex_Solver::add_obj_func_to_tableau(const int& num_rows,
                                                    const int& num_cols,
                                                    float** tableau,
                                                    Simplex_Problem& problem)
{
	int row = 0;
	int col = 0;
	for (Simplex_Problem::variable_iterator iter = problem.get_variable_iterator();
	     iter != problem.get_variable_end();
	     iter++)
	{
		const std::string& variable = *iter;
		float coeff = problem.get_obj_coeff(variable);
		if (coeff != 0) {
			tableau[row][col] = -1*coeff;
		}
		col++;
	}
}

//--------------------------------------------------------------------------
// ADD_CONSTRAINTS_TO_TABLEAU

void Cuda_Simplex_Solver::add_constraints_to_tableau(const int& num_rows,
                                                       const int& num_cols,
                                                       float** tableau,
                                                       Simplex_Problem& problem)
{
	int row = 1;
	for (Simplex_Problem::constraint_name_iterator i = problem.get_constraint_name_iterator();
	     i != problem.get_constraint_name_end();
	     i++)
	{
		int col = 0;
		const std::string& constraint_name = *i;
		Constraint& constraint = problem.get_constraint(constraint_name);
		constraint_type type = constraint.get_type();
		for (Simplex_Problem::variable_iterator j = problem.get_variable_iterator();
		     j != problem.get_variable_end();
		     j++)
		{
			const std::string& variable = *j;
			float coeff = constraint.get_coefficient(variable);
			if (coeff != 0) {
				// This is a = or <= constraint.
				if (type == LEQ || type == EQ) {
					tableau[row][col] = coeff;
				}
				// This is a >= constraint, we multiply by -1 to make it <=.
				else {
					tableau[row][col] = -1*coeff;
				}
			}

			// Move to the next column/variable.
			col++;
		}
		// Add the slack variable term.
		tableau[row][col+row-1] = 1;

		// Add the right hand side of the equatioin.
		float rhs = constraint.get_rhs();
		if (type == LEQ || type == EQ) {
			tableau[row][num_cols-1] = rhs;
		} else {
			tableau[row][num_cols-1] = -1*rhs;
		}

		// Move to the next constraint.
		row++;
	}
}
