////////////////////////////////////////////////////////////////////////////
//                            cmd_line_parser.h                           //
////////////////////////////////////////////////////////////////////////////

#ifndef CMD_LINE_PARSER_H
#define CMD_LINE_PARSER_H

#include "cmd_line_options.h"

class CMD_Line_Parser
{
public:
	// constructors and destructor
	CMD_Line_Parser(void);
	~CMD_Line_Parser(void);

	// methods
	CMD_Line_Options parse(int& argc, char** argv);
};

#endif
