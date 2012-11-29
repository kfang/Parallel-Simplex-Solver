/*
 * omp_simplex_solver.h
 *
 *  Created on: Nov 26, 2012
 *      Author: akaiser
 */

#ifndef OMP_SIMPLEX_SOLVER_H_
#define OMP_SIMPLEX_SOLVER_H_

#include "simplex_solver.h"
#include "simplex_problem.h"
#include "simplex_solution.h"

//--------------------------------------------------------------------------
// ompL_SIMPLEX_SOLVER

class Omp_Simplex_Solver : public Simplex_Solver
{
public:
	// Constructors and destructor.
	Omp_Simplex_Solver(void);
	~Omp_Simplex_Solver(void);

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
	           float** tableau);
	void get_pivot_col(int num_cols, int& pivot_col, float** tableau,
			float& min_val);
	void get_pivot_row(int num_rows, int pivot_col, int num_cols, int& pivot_row,
			float** tableau);
	void print_info(bool want_to_print, int num_rows, int num_cols, int pivot_row, int pivot_col,
			float** tableau);
	void print_results(int num_cols, double time, int num_variables, int var,
			float** tableau);
};


#endif /* OMP_SIMPLEX_SOLVER_H_ */
