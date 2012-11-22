////////////////////////////////////////////////////////////////////////////
//                            mps_file_parser.h                           //
////////////////////////////////////////////////////////////////////////////

#ifndef MPS_FILE_PARSER_H
#define MPS_FILE_PARSER_H

#include "input_file_parser.h"

//--------------------------------------------------------------------------
// MPS_FILE_PARSER

class MPS_File_Parser : public Input_File_Parser
{
public:
	// Constructor and destructor.
	MPS_File_Parser(void);
	~MPS_File_Parser(void);

	// methods
	Simplex_Problem parse(char* input_file);
};

#endif
