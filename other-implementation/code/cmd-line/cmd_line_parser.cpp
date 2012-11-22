////////////////////////////////////////////////////////////////////////////
//                           cmd_line_parser.cpp                          //
////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>

#include "cmd_line_parser.h"
#include "cmd_line_options.h"

//--------------------------------------------------------------------------
// CONSTRUCTOR AND DESTRUCTOR

CMD_Line_Parser::CMD_Line_Parser(void)
{
}

CMD_Line_Parser::~CMD_Line_Parser(void)
{
}

//--------------------------------------------------------------------------
// PRINT_HELP_MESSAGE

void print_help_message(void)
{
	printf("Usage: ./simplex [options] <problem files>\n\n");
	printf("Options:\n");
	printf("  %-27s%s\n", "-h, --help", "show this help message and exit");
	printf("  %-27s%s\n", "--debug", "run the program in debug mode");
	printf("  %-27s%s\n", "-c FILE, --config FILE", "give a configuration file that configures the solver");
}

//--------------------------------------------------------------------------
// PARSE

CMD_Line_Options CMD_Line_Parser::parse(int& argc, char** argv)
{
	// Default values for the options.
	char* config_file = 0;
	bool debug_mode = false;
	std::vector<char*> input_files;

	// Loop over each command line argument.
	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		// Help options
		if (arg == "-h" || arg == "--help") {
			// Print help message and the exit.
			print_help_message();
			exit(EXIT_SUCCESS);
		}

		// Debug option
		else if (arg == "--debug") {
			// Turn on debug mode for the program.
			debug_mode = true;
		}

		// Configuration file options
		else if (arg == "-c" || arg == "--config") {
			// The next file over is the configuration file.
			i++;
			if (i >= argc) {
				std::cerr << "ERROR: " << arg
				          << " option used but no configuration file given."
				          << std::endl;
				exit(EXIT_FAILURE);
			}
			config_file = argv[i];
		}

		// Input MPS file
		else {
			input_files.push_back(argv[i]);
		}
	}

	// Return a CMD_Line_Options argument with the correct settings.
	return CMD_Line_Options(config_file, debug_mode, input_files);
}
