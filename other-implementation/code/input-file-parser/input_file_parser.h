////////////////////////////////////////////////////////////////////////////
//                          input_file_parser.h                           //
////////////////////////////////////////////////////////////////////////////

#ifndef INPUT_FILE_PARSER_H
#define INPUT_FILE_PARSER_H

#include "configuration_options.h"
#include "simplex_problem.h"

//--------------------------------------------------------------------------
// INPUT_FILE_PARSER
class Input_File_Parser
{
public:
	virtual Simplex_Problem parse(char* input_file) = 0;

protected:
	Input_File_Parser(void) {}
};

//--------------------------------------------------------------------------
// CREATE_INPUT_FILE_PARSER

// Factory method for creating input file parsers.
Input_File_Parser* create_input_file_parser(Configuration_Options& config_opts);

#endif
