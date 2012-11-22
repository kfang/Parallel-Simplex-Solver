////////////////////////////////////////////////////////////////////////////
//                          input_file_parser.h                           //
////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <cstdlib>
#include "input_file_parser.h"
#include "configuration_options.h"
#include "mps_file_parser.h"

//--------------------------------------------------------------------------
// CREATE_INPUT_FILE_PARSER

// Factory method for creating input file parsers.
Input_File_Parser* create_input_file_parser(Configuration_Options& config_opts)
{
	// Create the parser based on the "problem_file_format" option in the
	// configuration file.  If none was specified use the mps type as
	// a default.
	std::string problem_file_format = config_opts.option_is_set("problem_file_format") ?
	                                  config_opts.get_option("problem_file_format") :
	                                  "mps";
	Input_File_Parser* parser;

	// Build an MPS parser.
	if (problem_file_format == "mps") {
		parser = new MPS_File_Parser();
	}

	// Problem file format does not exist.
	else {
		std::cerr << "ERROR: Problem file format not recognized."
			  << std::endl;
		exit(EXIT_FAILURE);
	}

	// Return the parser.
	return parser;
}
