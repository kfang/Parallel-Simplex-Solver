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

	// Calculate the number of rows and columns in the tableau and allocate memory.
	int num_rows = num_constraints + 1;
	int num_cols = num_variables + num_constraints + 1;

	// While the objective function can be increased, find a better
	// vertex on the simplex.
	int pivot_col, pivot_row;
	for (;;) {
		for (pivot_col = 0; (pivot_col < num_cols-1) && (tableau[0][pivot_col] >= 0); pivot_col++);
		for (pivot_row = 1; (pivot_row < num_rows) && (tableau[pivot_row][pivot_col] <= 0); pivot_row++);
		if (pivot_row >= num_rows || pivot_col >= num_cols-1) {
			std::cout << "pivot_row: " << pivot_row << std::endl;
			std::cout << "pivot_col: " << pivot_col << std::endl;
			break;
		}
		for (int i = pivot_row+1; i < num_rows; i++)
			if (tableau[i][pivot_col] > 0)
				if (tableau[i][num_cols-1]/tableau[i][pivot_col] < tableau[pivot_row][num_cols-1]/tableau[pivot_row][pivot_col])
					pivot_row = i;
		// Check for unboundedness
		for (int i = 0; i < num_cols; i++) {
			for (int j = 0; j < num_rows; j++) {
				if (tableau[j][i] < 0) {
					if (j == num_rows-1) {
						//Then unbounded
						std::cout << "The problem is unbounded\n";
						return Simplex_Solution();
					}
				} else {
					break;
				}
			}
		}
		std::cout << "---------------------------------" << std::endl;
		std::cout << "BEFORE PIVOT" << std::endl;
		print_matrix(num_rows, num_cols, tableau);
		std::cout << "pivot_row: " << pivot_row << std::endl;
		std::cout << "pivot_col: " << pivot_col << std::endl;
		std::cout << "AFTER PIVOT" << std::endl;
		pivot(pivot_row, pivot_col, num_rows, num_cols, tableau);
		print_matrix(num_rows, num_cols, tableau);
	}

	std::cout << "DONE!!!" << std::endl;

	return Simplex_Solution();
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
