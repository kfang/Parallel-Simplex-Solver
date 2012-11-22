////////////////////////////////////////////////////////////////////////////
//                          cmd_line_options.cpp                          //
////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <vector>
#include "cmd_line_options.h"

//--------------------------------------------------------------------------
// CONSTRUCTOR AND DESTRUCTOR

CMD_Line_Options::CMD_Line_Options(char* config_file,
                                   bool debug_mode,
                                   std::vector<char*> input_files)
{
	this->config_file = config_file;
	this->debug_mode = debug_mode;
	this->input_files = input_files;
}

CMD_Line_Options::~CMD_Line_Options(void)
{
}

//--------------------------------------------------------------------------
// PRINT

void CMD_Line_Options::print(void)
{
	std::cout << "config_file : " << (config_file == 0 ? "NULL" : config_file) << std::endl;
	std::cout << "debug_mode  : " << (debug_mode ? "true" : "false") << std::endl;
	std::cout << "input_files : [";
	for (int i = 0; i < input_files.size(); i++) {
		if (i != 0) {
			std::cout << ", ";
		}
		std::cout << input_files[i];
	}
	std::cout << "]" << std::endl;
}
