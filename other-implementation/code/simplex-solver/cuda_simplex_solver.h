/*
 * cuda_simplex_solver.h
 *
 *  Created on: Nov 25, 2012
 *      Author: akaiser
 */

#ifndef CUDA_SIMPLEX_SOLVER_H_
#define CUDA_SIMPLEX_SOLVER_H_

#include "simplex_solver.h"
#include "simplex_problem.h"
#include "simplex_solution.h"

//--------------------------------------------------------------------------
// CUDA_SIMPLEX_SOLVER

class Cuda_Simplex_Solver : public Simplex_Solver
{
public:
	// Constructors and destructor.
	Cuda_Simplex_Solver(void);
	~Cuda_Simplex_Solver(void);

	// methods
	Simplex_Solution solve(Simplex_Problem& problem);

private:
	float** create_tableau(Simplex_Problem& problem);
	void add_obj_func_to_tableau(const int& num_rows,
	                             const int& num_cols,
	                             float** tableau,
	                             Simplex_Problem& problem);
	void add_constraints_to_tableau(const int& num_rows,
	                                const int& num_cols,
	                                float** tableau,
	                                Simplex_Problem& problem);
	void pivot(const int& pivot_row, const int& pivot_col,
	           const int& num_rows, const int& num_cols,
	           float** tableau, float** cuda_tableau);
};

__global__ void cuda_pivot(int pivot_row, int pivot_col,
		int num_rows, int num_cols,
		float** tableau);

#endif /* CUDA_SIMPLEX_SOLVER_H_ */
