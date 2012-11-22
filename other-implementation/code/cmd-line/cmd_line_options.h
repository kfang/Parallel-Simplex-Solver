////////////////////////////////////////////////////////////////////////////
//                             cmd_line_options.h                         //
////////////////////////////////////////////////////////////////////////////

#ifndef CMD_LINE_OPTIONS_H
#define CMD_LINE_OPTIONS_H

#include <vector>

class CMD_Line_Options
{
public:
	// constructors and destructor
	CMD_Line_Options(char* config_file,
	                 bool debug_mode,
	                 std::vector<char*> input_files);
	~CMD_Line_Options(void);

	// methods
	void print(void);

	char* config_file;
	std::vector<char*> input_files;
	bool debug_mode;
};

#endif
