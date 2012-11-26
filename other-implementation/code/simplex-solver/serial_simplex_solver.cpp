////////////////////////////////////////////////////////////////////////////
//                       serial_simplex_solver.cpp                        //
////////////////////////////////////////////////////////////////////////////


#include <iostream>
#include <cstdio>
#include <string>
#include "serial_simplex_solver.h"
#include "simplex_problem.h"
#include "simplex_solution.h"
#include "util.h"

//--------------------------------------------------------------------------
// CONSTRUCTORS AND DESTRUCTOR

Serial_Simplex_Solver::Serial_Simplex_Solver(void)
{
}

Serial_Simplex_Solver::~Serial_Simplex_Solver(void)
{
}

//--------------------------------------------------------------------------
// SOLVE

Simplex_Solution Serial_Simplex_Solver::solve(Simplex_Problem& problem)
{
	// Make a new tableau for solving the problem.
	float** tableau = create_tableau(problem);

	// Get the number of variables and constraints in the problem.
	int num_variables = problem.get_num_variables();
	int num_constraints = problem.get_num_constraints();

	// Calculate the number of rows and columns in the tableau.
	int num_rows = num_constraints + 1;
	int num_cols = num_variables + num_constraints + 1;

	// While the objective function can be increased, find a better
	// vertex on the simplex.  This section will throw an exception
	// when it stops iterating over the vertices.
	double start_time = timestamp();
	int num_iterations = 0;
	Simplex_Solution solution;
	try {
		while (true) {
			num_iterations++;
			find_better_vertex(num_rows, num_cols, tableau);
		}
	}

	// The objective function can no longer be improved.  Create a solution
	// object with all necessary information about the running of the algorithm.
	catch (solution_type type) {
		double end_time = timestamp();
		double run_time = end_time - start_time;
		solution.set_name(problem.get_name() + " Solution");
		solution.set_type(type);
		solution.set_num_variables(problem.get_num_variables());
		solution.set_num_constraints(problem.get_num_constraints());
		solution.set_run_time(run_time);
		solution.set_obj_func_val(tableau[0][num_cols-1]);
		solution.set_num_iterations(num_iterations);
		// TODO: fill out variable coefficients.
	}

	// Clean up the memory used for the tableau.
	delete_2D_array(num_rows, num_cols, tableau);

	return solution;
}

//--------------------------------------------------------------------------
// FIND_BETTER_VERTEX

void Serial_Simplex_Solver::find_better_vertex(const int& num_rows,
                                               const int& num_cols,
                                               float** tableau)
{
	int pivot_col = choose_pivot_column(num_rows, num_cols, tableau);
	int pivot_row = choose_pivot_row(pivot_col, num_rows, num_cols, tableau);
	pivot(pivot_row, pivot_col, num_rows, num_cols, tableau);
}

//--------------------------------------------------------------------------
// CHOOSE_PIVOT_COLUMN

int Serial_Simplex_Solver::choose_pivot_column(const int& num_rows,
                                               const int& num_cols,
                                               float** tableau)
{
	// Set the pivot column to be the first negative entry of the objective function.
	int pivot_col = 0;
	while ((pivot_col < num_cols-1) && (tableau[0][pivot_col] >= 0)) {
	    pivot_col++;
	}

	// If there were no negative entries in the objective funtion then it can
	// no longer be increased/
	if (pivot_col >= num_cols-1) {
		throw OPTIMAL_SOLUTION_FOUND;
	}

	return pivot_col;
}

//--------------------------------------------------------------------------
// CHOOSE_PIVOT_ROW

int Serial_Simplex_Solver::choose_pivot_row(const int& pivot_col,
                                            const int& num_rows,
                                            const int& num_cols,
                                            float** tableau)
{
	// Initialize the pivot row to be the first non-zero, non-negative
	// entry in the pivot column.
	int pivot_row = 1;
	while ((pivot_row < num_rows) && (tableau[pivot_row][pivot_col] <= 0)) {
		pivot_row++;
	}

	// If there are no positive, non-zero entries in the row then we
	// know we have an unbouneded problem.
	if (pivot_row >= num_rows) {
		throw UNBOUNDED_SOLUTION;
	}

	// Update the pivot row if we find a row with smaller ratio.
	for (int row = pivot_row+1; row < num_rows; row++) {
		if (tableau[row][pivot_col] > 0) {
			float pivot_row_ratio = tableau[pivot_row][num_cols-1]/tableau[pivot_row][pivot_col];
			float curr_row_ratio = tableau[row][num_cols-1]/tableau[row][pivot_col];
			if (curr_row_ratio < pivot_row_ratio) {
				pivot_row = row;
			}
		}
	}

	return pivot_row;
}

//--------------------------------------------------------------------------
// PIVOT

void Serial_Simplex_Solver::pivot(const int& pivot_row, const int& pivot_col,
                            const int& num_rows, const int& num_cols,
                            float** tableau)
{
	// Keep the pivot value in a register.
	float pivot_val = tableau[pivot_row][pivot_col];

	// Zero out the column above and below the pivot.
	for (int row = 0; row < num_rows; row++) {
		float scale = tableau[row][pivot_col]/pivot_val;
		if (row != pivot_row) {
			for (int col = 0; col < num_cols; col++) {
				tableau[row][col] -= scale*tableau[pivot_row][col];
			}
		}
	}

	// Scale the pivot row.
	for (int col = 0; col < num_cols; col++) {
		tableau[pivot_row][col] /= pivot_val;
	}
}

//--------------------------------------------------------------------------
// CREATE_TABLEAU

float** Serial_Simplex_Solver::create_tableau(Simplex_Problem& problem)
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

void Serial_Simplex_Solver::add_obj_func_to_tableau(const int& num_rows,
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

void Serial_Simplex_Solver::add_constraints_to_tableau(const int& num_rows,
                                                       const int& num_cols,
                                                       float** tableau,
                                                       Simplex_Problem& problem)
{
	// TODO: This needs to be cleaned up and made more readable.  It should
	// detect problems that don't have the origin as a solution and report it.
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
