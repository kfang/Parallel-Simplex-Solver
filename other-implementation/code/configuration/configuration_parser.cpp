////////////////////////////////////////////////////////////////////////////
//                        configuration_parser.cpp                        //
////////////////////////////////////////////////////////////////////////////

#include <cstdlib>
#include <fstream>
#include <boost/regex.hpp>

#include "configuration_parser.h"
#include "configuration_options.h"

//--------------------------------------------------------------------------
// CONSTRUCTORS AND DESTRUCTORS

Configuration_Parser::Configuration_Parser(void)
{
}

Configuration_Parser::~Configuration_Parser(void)
{
}

//--------------------------------------------------------------------------
// PARSE

Configuration_Options Configuration_Parser::parse(const char* config_file)
{
	std::ifstream input_file(config_file);
	Configuration_Options options;

	// regex for the file format
	boost::cmatch matches;
	boost::regex blank_line("\\s*|#.*");
	boost::regex option_line("\\s*(\\w+)\\s*=\\s*(\\w+)\\s*(?:#.*)?");

	int line_num = 1;
	std::string next_line;
	while (std::getline(input_file, next_line)) {
		// Check if it is a blank line.
		if (boost::regex_match(next_line.c_str(),
		                       matches,
		                       blank_line))
		{
			// Skip this line, it's blank or a comment.
		}

		// Check if it is an option line.
		else if (boost::regex_match(next_line.c_str(),
		                       matches,
		                       option_line))
		{
			std::string option = matches[1].str();
			std::string value = matches[2].str();
			options.set_option(option, value);
		}

		// If we get here, we have an invalid line.
		else {
			// invalid line
			std::cerr << "ERROR: Invalid line in the configuration file at line "
			          << line_num << std::endl;
			std::cerr << next_line << std::endl;
			exit(EXIT_FAILURE);
		}

		// Update the line number.
		line_num++;
	}

	return options;
}
