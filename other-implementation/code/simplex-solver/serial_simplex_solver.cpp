////////////////////////////////////////////////////////////////////////////
//                       serial_simplex_solver.cpp                        //
////////////////////////////////////////////////////////////////////////////


#include <iostream>
#include "serial_simplex_solver.h"
#include "simplex_problem.h"
#include "simplex_solution.h"


//--------------------------------------------------------------------------
// CONSTRUCTORS AND DESTRUCTOR

Serial_Simplex_Solver::Serial_Simplex_Solver(void)
{
	// TODO
}

Serial_Simplex_Solver::~Serial_Simplex_Solver(void)
{
}

//--------------------------------------------------------------------------
// SOLVE

Simplex_Solution Serial_Simplex_Solver::solve(Simplex_Problem& problem)
{
	// TODO
	std::cout << "Inside Serial_Simplex_Solver::solve" << std::endl;
	return Simplex_Solution();
}
