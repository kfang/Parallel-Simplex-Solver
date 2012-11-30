////////////////////////////////////////////////////////////////////////////
//                        omp_simplex_solver.cpp                          //
////////////////////////////////////////////////////////////////////////////


#include <iostream>
#include <cstdio>
#include <string>
#include <string.h>
#include "omp_simplex_solver.h"
#include "simplex_problem.h"
#include "simplex_solution.h"
#include "util.h"
#include "omp.h"

//--------------------------------------------------------------------------
// CONSTRUCTORS AND DESTRUCTOR

Omp_Simplex_Solver::Omp_Simplex_Solver(void)
{
}

Omp_Simplex_Solver::~Omp_Simplex_Solver(void)
{
}


//--------------------------------------------------------------------------
// SOLVE

Simplex_Solution Omp_Simplex_Solver::solve(Simplex_Problem& problem)
{
	for (int var = 0; var < 3; var++) {
		omp_set_num_threads(4);
		double time = timestamp();

		// Make a new tableau for solving the problem.
		float** tableau = create_tableau(problem);

		// Get the number of variables and constraints in the problem.
		int num_variables = problem.get_num_variables();
		int num_constraints = problem.get_num_constraints();

		// Calculate the number of rows and columns in the tableau and allocate memory.
		int num_rows = num_constraints + 1;
		int num_cols = num_variables + num_constraints + 1;
		time = timestamp() - time;
		std::cerr << "Setup time: " << time << std::endl;
		time = timestamp();

		// While the objective function can be increased, find a better
		// vertex on the simplex.
		int pivot_col, pivot_row;
		float min_val;
		for (;;) {
			min_val = tableau[0][0];
			get_pivot_col(num_cols, pivot_col, tableau, min_val);
			for (pivot_row = 1; (pivot_row < num_rows) && (tableau[pivot_row][pivot_col] <= 0); pivot_row++);
			if (min_val >= 0) {
				break;
			}
			if (pivot_row >= num_rows) {
				//Then unbounded
				std::cout << "The problem is unbounded\n";
				return Simplex_Solution();
			}
			get_pivot_row(num_rows, pivot_col, num_cols, pivot_row,tableau);
			print_info(false, num_rows, num_cols, pivot_row, pivot_col, tableau);
			pivot(pivot_row, pivot_col, num_rows, num_cols, tableau);
			//print_matrix(num_rows, num_cols, tableau);
		}

		print_results(num_cols, time, num_variables, var, tableau);
	}

	return Simplex_Solution();
}

//--------------------------------------------------------------------------
// PIVOT

void Omp_Simplex_Solver::pivot(const int& pivot_row, const int& pivot_col,
                            const int& num_rows, const int& num_cols,
                            float** tableau)
{
	// Keep the pivot value in a register.
	float pivot_val = tableau[pivot_row][pivot_col];

	// Zero out the column above and below the pivot.
#pragma omp parallel
	{
		#pragma omp for
		for (int row = 0; row < num_rows; row++) {
			float scale = tableau[row][pivot_col]/pivot_val;
			if (row != pivot_row) {
				for (int col = 0; col < num_cols; col++) {
					tableau[row][col] -= scale*tableau[pivot_row][col];
				}

				//----Tried implementing Duff's Device
				// int runs = (num_cols + 2)/3;
				// int col = 0;
				// switch(num_cols % 3)
				// {
				// 	case 0: do { tableau[row][col++] -= scale*tableau[pivot_row][col];
				// 	case 2: 	 tableau[row][col++] -= scale*tableau[pivot_row][col];
				// 	case 1: 	 tableau[row][col++] -= scale*tableau[pivot_row][col];
				// 	} while (--runs >0);
				// }
			}
		}
	

		// Scale the pivot row.
		#pragma omp for
		for (int col = 0; col < num_cols; col++) {
			tableau[pivot_row][col] /= pivot_val;
		}
	}
}

//--------------------------------------------------------------------------
// Get pivot row and column

void Omp_Simplex_Solver::get_pivot_col(int num_cols, int& pivot_col,
		float** tableau, float& min_val) {
	pivot_col = 0;
	for (int i = 0; (i < num_cols - 1); i++) {
		if (tableau[0][i] < min_val) {
			min_val = tableau[0][i];
			pivot_col = i;
		}
	}

	// -----Parallel find min-----------
	// float shared_min;
	// int shared_piv_col;
	// #pragma omp parallel 
	// {
	// 	float min = min_val;
	// 	int piv_col = 0;

	// 	#pragma omp for nowait
	// 	for (int i = 0; i < (num_cols - 1); i++){
	// 		if (tableau[0][i] < min){
	// 			min = tableau[0][i];
	// 			piv_col = i;
	// 		}
	// 	}

	// 	#pragma omp critical 
	// 	{
	// 		if (min < shared_min){
	// 			shared_min = min;
	// 			shared_piv_col = piv_col;
	// 		}
	// 	}
	// }

	// min_val = shared_min;
	// pivot_col = shared_piv_col;
}

void Omp_Simplex_Solver::get_pivot_row(int num_rows, int pivot_col, int num_cols,
		int& pivot_row, float** tableau) {
	// for (int i = pivot_row + 1; i < num_rows; i++)
	// 	if (tableau[i][pivot_col] > 0)
	// 		if (tableau[i][num_cols - 1] / tableau[i][pivot_col] < tableau[pivot_row][num_cols - 1]/tableau[pivot_row][pivot_col])
	// 			pivot_row = i;


	//---storing ratio explicitly---
	float minRatio = tableau[pivot_row][num_cols - 1]/tableau[pivot_row][pivot_col];
	for (int i = pivot_row + 1 ; i < num_rows; i++){
		if (tableau[i][pivot_col] > 0){
			float rowRatio = tableau[i][num_cols - 1] / tableau[i][pivot_col];
			if (rowRatio < minRatio){
				pivot_row = i;
				minRatio = rowRatio;
			}
		}
	}
}


//--------------------------------------------------------------------------
// CREATE_TABLEAU

float** Omp_Simplex_Solver::create_tableau(Simplex_Problem& problem)
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

void Omp_Simplex_Solver::add_obj_func_to_tableau(const int& num_rows,
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

void Omp_Simplex_Solver::add_constraints_to_tableau(const int& num_rows,
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

void Omp_Simplex_Solver::print_info(bool want_to_print, int num_rows, int num_cols, int pivot_row,
		int pivot_col, float** tableau) {
	if (want_to_print) {
		std::cerr << "---------------------------------" << std::endl;
		std::cerr << "BEFORE PIVOT" << std::endl;
		print_matrix(num_rows, num_cols, tableau);
		std::cerr << "pivot_row: " << pivot_row << std::endl;
		std::cerr << "pivot_col: " << pivot_col << std::endl;
		std::cerr << "AFTER PIVOT" << std::endl;
	}
}

void Omp_Simplex_Solver::print_results(int num_cols, double time,
		int num_variables, int var, float** tableau) {
	std::cerr << "DONE!!!" << std::endl;
	std::cerr << "Max value: " << tableau[0][num_cols - 1] << std::endl;
	time = timestamp() - time;
	std::cerr << "Solve time: " << time << std::endl;
	std::cout << num_variables << "," << time << "," << var << std::endl;
}
