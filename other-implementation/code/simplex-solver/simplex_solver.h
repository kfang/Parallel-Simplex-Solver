////////////////////////////////////////////////////////////////////////////
//                            simplex_solver.h                            //
////////////////////////////////////////////////////////////////////////////

#ifndef SIMPLEX_SOLVER_H
#define SIMPLEX_SOLVER_H

#include <iostream>
#include <cstdlib>
#include <string>
#include "configuration_options.h"
#include "simplex_problem.h"
#include "simplex_solution.h"

//--------------------------------------------------------------------------
// SIMPLEX_SOLVER

class Simplex_Solver
{
public:
	virtual Simplex_Solution solve(Simplex_Problem& problem) = 0;

protected:
	Simplex_Solver(void) {}
};

//--------------------------------------------------------------------------
// CREATE_SOLVER

// Factory method for creating simplex solvers.
Simplex_Solver* create_solver(Configuration_Options& config_opts);

#endif
