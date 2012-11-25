////////////////////////////////////////////////////////////////////////////
//                                 main.cpp                               //
////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cstdio>
#include <string>

#include "cmd_line_parser.h"
#include "cmd_line_options.h"
#include "configuration_parser.h"
#include "configuration_options.h"
#include "input_file_parser.h"
#include "mps_file_parser.h"
#include "simplex_problem.h"
#include "simplex_solution.h"
#include "simplex_solver.h"

int main(int argc, char** argv)
{
	// Parse the command line to get the command line options and input files.
	CMD_Line_Options cmd_line_opts = CMD_Line_Parser().parse(argc, argv);

	// Parse the configuration file to get the configuration options.
	const char* config_file = cmd_line_opts.config_file;
	Configuration_Options config_opts = Configuration_Parser().parse(config_file);

	// Create a solver using the specified options.
	Simplex_Solver* solver = create_solver(config_opts);

	// Create an input file parser using the specified options.
	Input_File_Parser* input_file_parser = create_input_file_parser(config_opts);

	// For each input file create a problem instance, and find a solution.
	for (int i = 0; i < cmd_line_opts.input_files.size(); i++) {
		// Parse the current input file and build a problem instance.
		char* input_file = cmd_line_opts.input_files[i];
		Simplex_Problem problem = input_file_parser->parse(input_file);
		std::cout << "Problem is in parsing\n";
		// TODO get rid of this print statement
		problem.print();

		// Find a solution to the problem.
		Simplex_Solution solution = solver->solve(problem);
/*

		// Output the solution and statistics regarding how long it took to
		// solve, number of iterations, etc.
		cout << solution.get_statistics() << endl;
*/
	}


	// Free the memory allocated for the solver.
	delete solver;
	delete input_file_parser;
}
