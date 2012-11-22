////////////////////////////////////////////////////////////////////////////
//                        configuration_options.cpp                       //
////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <map>

#include "configuration_options.h"

//--------------------------------------------------------------------------
// CONSTRUCTOR AND DESTRUCTOR

Configuration_Options::Configuration_Options(void)
{
}

Configuration_Options::~Configuration_Options(void)
{
}

//--------------------------------------------------------------------------
// OPTION_IS_SET

bool Configuration_Options::option_is_set(std::string option)
{
	return options.find(option) != options.end();
}

//--------------------------------------------------------------------------
// SET_OPTION

void Configuration_Options::set_option(std::string option, std::string val)
{
	options[option] = val;
}

//--------------------------------------------------------------------------
// GET_OPTION

std::string Configuration_Options::get_option(std::string option)
{
	return options[option];
}

//--------------------------------------------------------------------------
// PRINT

void Configuration_Options::print(void)
{
	std::cout << "{" << std::endl;
	for (std::map<std::string, std::string>::iterator iter = options.begin();
	     iter != options.end();
	     iter++)
	{
		std::string option = iter->first;
		std::string val = options[option];
		std::cout << "\t" << option << " : " << val << std::endl;
	}
	std::cout << "}" << std::endl;
}
