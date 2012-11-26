////////////////////////////////////////////////////////////////////////////
//                           simplex_solver.cpp                           //
////////////////////////////////////////////////////////////////////////////


#include <iostream>
#include <cstdlib>
#include <string>
#include "simplex_solver.h"
#include "configuration_options.h"
#include "simplex_problem.h"
#include "simplex_solution.h"
#include "serial_simplex_solver.h"
#include "cuda_simplex_solver.h"

//--------------------------------------------------------------------------
// CREATE_SOLVER

// Factory method for creating simplex solvers.
Simplex_Solver* create_solver(Configuration_Options& config_opts)
{

	// Create the solver based on the "solver" option in the configuration
	// file.  If none was specified use the serial type as a default.
	std::string solver_type = config_opts.option_is_set("solver") ?
	                          config_opts.get_option("solver") :
	                          "serial";
	Simplex_Solver* solver;

	// Build a serial solver.
	if (solver_type == "serial") {
		solver = new Serial_Simplex_Solver();
	}

	// Build a CUDA solver.
	else if (solver_type == "cuda") {
		solver = new Cuda_Simplex_Solver();
	}

	// Build an OMP solver.
	else if (solver_type == "omp") {
		solver = new Omp_Simplex_Solver();
	}

	// Solver type does not exist.
	else {
		std::cerr << "ERROR: Solver type does not exist."
			  << std::endl;
		exit(EXIT_FAILURE);
	}

	// Return the simplex solver.
	return solver;
}
