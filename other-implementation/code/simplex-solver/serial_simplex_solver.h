////////////////////////////////////////////////////////////////////////////
//                        serial_simplex_solver.h                         //
////////////////////////////////////////////////////////////////////////////

#ifndef SERIAL_SIMPLEX_SOLVER_H
#define SERIAL_SIMPLEX_SOLVER_H

#include "simplex_solver.h"
#include "simplex_problem.h"
#include "simplex_solution.h"

//--------------------------------------------------------------------------
// SERIAL_SIMPLEX_SOLVER

class Serial_Simplex_Solver : public Simplex_Solver
{
public:
	// Constructors and destructor.
	Serial_Simplex_Solver(void);
	~Serial_Simplex_Solver(void);

	// methods
	Simplex_Solution solve(Simplex_Problem& problem);

private:
	// TODO: Put tableau and stuff here.
};

#endif
