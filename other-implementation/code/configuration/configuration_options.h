////////////////////////////////////////////////////////////////////////////
//                          configuration_options.h                       //
////////////////////////////////////////////////////////////////////////////

#ifndef CONFIGURATION_OPTIONS_H
#define CONFIGURATION_OPTIONS_H

#include <string>
#include <map>

class Configuration_Options
{
public:
	// constructors and destructor
	Configuration_Options(void);
	~Configuration_Options(void);

	// methods
	bool option_is_set(std::string option);
	void set_option(std::string option, std::string val);
	std::string get_option(std::string option);
	void print(void);

private:
	std::map<std::string, std::string> options;
};

#endif
