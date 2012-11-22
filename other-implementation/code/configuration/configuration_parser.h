////////////////////////////////////////////////////////////////////////////
//                         configuration_parser.h                         //
////////////////////////////////////////////////////////////////////////////

#ifndef CONFIGURATION_PARSER_H
#define CONFIGURATION_PARSER_H

#include "configuration_options.h"

class Configuration_Parser
{
public:
	// constructors and destructor
	Configuration_Parser(void);
	~Configuration_Parser(void);

	// methods
	Configuration_Options parse(const char* config_file);
};

#endif
