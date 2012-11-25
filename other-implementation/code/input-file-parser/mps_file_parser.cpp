////////////////////////////////////////////////////////////////////////////
//                           mps_file_parser.cpp                          //
////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <boost/regex.hpp>
#include <map>
#include <cstdlib>
#include "mps_file_parser.h"
#include "simplex_problem.h"
#include <sstream>

//--------------------------------------------------------------------------
// DEFINITIONS

// States the parser can be in.
enum state { EXPECTING_NAME,
             EXPECTING_ROW_START,
             IN_ROWS_SECTION,
             IN_COLUMNS_SECTION,
             IN_RHS_SECTION,
             IN_BOUNDS_SECTION,
             DONE };

// Regex for the line types in the mps file format.
const boost::regex blank_line("\\s*|\\*.*");
const boost::regex name_line("\\s*NAME\\s+(\\w+)\\s*");
const boost::regex row_start("\\s*ROWS\\s*");
const boost::regex row_line("\\s*(E|L|G|N)\\s*(\\w+)\\s*");
const boost::regex col_start("\\s*COLUMNS\\s*");
const boost::regex col_line("\\s*(\\w+)(?:\\s+(\\w+)\\s+([0-9.+-eE]+)(?:\\s+(\\w+)\\s+([0-9.+-eE]+))?)?\\s*");
const boost::regex rhs_start("\\s*RHS\\s*");
const boost::regex rhs_line("\\s*(\\w+)\\s+(\\w+)\\s+([0-9.+-eE]+)(\\s+(\\w+)\\s+([0-9.+-eE]+))?\\s*");
const boost::regex bounds_start("\\s*BOUNDS\\s*");
const boost::regex bounds_line("\\s*(UP|LO|FX|FR)\\s*(\\w+)\\s*(\\w+)\\s*([0-9.+-eE]+)\\s*");
const boost::regex end_data("\\s*ENDATA\\s*");

//--------------------------------------------------------------------------
// HANDLE_NAME_LINE

void handle_name_line(const boost::cmatch& matched_line,
                      Simplex_Problem& problem_instance)
{
	std::string name = matched_line[1].str();
	problem_instance.set_name(name);
}

//--------------------------------------------------------------------------
// HANDLE_ROW_LINE

void handle_row_line(const boost::cmatch& matched_line,
                     Simplex_Problem& problem_instance)
{
	std::string type = matched_line[1].str();
	std::string name = matched_line[2].str();

	// This is an equality constraint.
	if (type == "E") {
		problem_instance.add_constraint(name, EQ);
	}

	// This is a less than equal to constraint.
	else if (type == "L") {
		problem_instance.add_constraint(name, LEQ);
	}

	// This is a greater than equal to constraint.
	else if (type == "G") {
		problem_instance.add_constraint(name, GEQ);
	}

	// This is the objective function.
	else if (type == "N") {
		// TODO: add a way to customize the optimization type.
		problem_instance.add_obj_func(name, MAX);
	}

	// This is an invalid constraint type.
	else {
		std::cerr << "ERROR: Invalid constraint type -> "
		          << type << std::endl;
		exit(EXIT_FAILURE);
	}
}

//--------------------------------------------------------------------------
// HANDLE_COL_LINE

void handle_col_line(const boost::cmatch& matched_line,
                     Simplex_Problem& problem_instance)
{
	// Name of the column this is describing.
	std::string col_name = matched_line[1].str();

	// This contains the name of the first row if it was specified.
	std::string row1_name = matched_line[2].str();
	bool row1_specified = row1_name != "";

	// This contains the name of the second row if it was specified.
	std::string row2_name = matched_line[4].str();
	bool row2_specified = row2_name != "";

	// Update the problem for the rows that were specified.
	if (row1_specified) {
		float coeff_val = atof(matched_line[3].str().c_str());

		// The line is modifying an entry in the obj function.
		if (problem_instance.is_obj_func(row1_name)) {
			problem_instance.set_obj_coeff(col_name, coeff_val);
		}

		// The line is modifying an entry in some constraint.
		else {
			problem_instance.set_constraint_coeff(row1_name, col_name, coeff_val);
		}
	}

	if (row2_specified) {
		float coeff_val = atof(matched_line[5].str().c_str());

		// The line is modifying an entry in the obj function.
		if (problem_instance.is_obj_func(row2_name)) {
			problem_instance.set_obj_coeff(col_name, coeff_val);
		}

		// The line is modifying an entry in some constraint.
		else {
			problem_instance.set_constraint_coeff(row2_name, col_name, coeff_val);
		}
	}
}

//--------------------------------------------------------------------------
// HANDLE_RHS_LINE

void handle_rhs_line(const boost::cmatch& matched_line,
                     Simplex_Problem& problem_instance)
{
	// This contains the name of the first row.
	std::string row1_name = matched_line[2].str();
	float row1_val = atof(matched_line[3].str().c_str());

	// Set the right hand side entry for the first row.
	problem_instance.set_constraint_rhs(row1_name, row1_val);

	// This contains the name of the second row if it was specified.
	std::string row2_name = matched_line[4].str();
	bool row2_specified = row2_name != "";

	// Set the right hand side entry for the second row if needed.
	if (row2_specified) {
		float row2_val = atof(matched_line[5].str().c_str());
		problem_instance.set_constraint_rhs(row1_name, row1_val);
	}
}

//--------------------------------------------------------------------------
// HANDLE_BOUNDS_LINE

void handle_bounds_line(const boost::cmatch& matched_line,
                     Simplex_Problem& problem_instance, int line_num)
{
	std::string type = matched_line[1].str();
	std::string col_name = matched_line[3].str();
	float val = atof(matched_line[4].str().c_str());
	std::stringstream ss;
	ss << "bound" << line_num;
	std::string bound_name = ss.str();
	std::cout << "Type: " << type << std::endl;
	std::cout << "Col_name: " << col_name << std::endl;
	std::cout << "Val: " << val << std::endl;
	std::cout << "bound_name: " << bound_name << std::endl;

	// This is an equality constraint.
	if (type == "LO") {
		problem_instance.add_constraint(bound_name, GEQ);
		problem_instance.set_constraint_coeff(bound_name, col_name, 1.0);
		problem_instance.set_constraint_rhs(bound_name, val);
	}

	// This is a less than equal to constraint.
	else if (type == "UP") {
		problem_instance.add_constraint(bound_name, LEQ);
		problem_instance.set_constraint_coeff(bound_name, col_name, 1.0);
		problem_instance.set_constraint_rhs(bound_name, val);
	}

	// This is a greater than equal to constraint.
	else if (type == "FX") {
		problem_instance.add_constraint(bound_name, EQ);
		problem_instance.set_constraint_coeff(bound_name, col_name, 1.0);
		problem_instance.set_constraint_rhs(bound_name, val);
	}

	// This is the objective function.
	else if (type == "FR") {
		//Don't do anything
	}

	// This is an invalid constraint type.
	else {
		std::cerr << "ERROR: Invalid constraint type -> "
		          << type << std::endl;
		exit(EXIT_FAILURE);
	}
}

//--------------------------------------------------------------------------
// CONSTRUCTOR AND DESTRUCTOR FOR MPS PARSER

MPS_File_Parser::MPS_File_Parser(void)
{
}

MPS_File_Parser::~MPS_File_Parser(void)
{
}

//--------------------------------------------------------------------------
// PARSE

Simplex_Problem MPS_File_Parser::parse(char* input_file_name)
{

	// Open the input file.
	std::ifstream input_file(input_file_name);

	// Start the parser of in the EXPECTING_NAME state.
	state curr_state = EXPECTING_NAME;

	// Create the problem instance.  The problem will be filled out
	// as data from the file is gathered.
	Simplex_Problem problem_instance;

	int line_num = 1;
	std::string next_line;
	boost::cmatch matches;
	while(std::getline(input_file, next_line) && curr_state != DONE) {
		// Check if it is a blank line.
		if (boost::regex_match(next_line.c_str(),
		                       matches,
		                       blank_line))
		{
			// Ignore this line, it is blank or a comment.
		}

		// Check if it is the end of the file.
		else if (boost::regex_match(next_line.c_str(),
		                            matches,
		                            end_data))
		{
			curr_state = DONE;
			break;
		}

		// Check if it is the name line.
		else if (boost::regex_match(next_line.c_str(),
		                            matches,
		                            name_line))
		{
			if (curr_state != EXPECTING_NAME) {
				std::cerr << "ERROR: at line " << line_num << std::endl
				          << next_line << std::endl;
				exit(EXIT_FAILURE);
			}

			handle_name_line(matches, problem_instance);
			curr_state = EXPECTING_ROW_START;
		}

		// Check if it is the start of the rows section.
		else if (boost::regex_match(next_line.c_str(),
		                            matches,
		                            row_start))
		{
			if (curr_state != EXPECTING_ROW_START) {
				std::cerr << "ERROR: at line " << line_num << std::endl
				          << next_line << std::endl;
				exit(EXIT_FAILURE);
			}

			curr_state = IN_ROWS_SECTION;
		}

		// Check if it is a row line.
		else if (boost::regex_match(next_line.c_str(),
		                            matches,
		                            row_line))
		{
			if (curr_state != IN_ROWS_SECTION) {
				std::cerr << "ERROR: at line " << line_num << std::endl
				          << next_line << std::endl;
				exit(EXIT_FAILURE);
			}

			handle_row_line(matches, problem_instance);
		}

		// Check if it is the start of the columns section.
		else if (boost::regex_match(next_line.c_str(),
		                            matches,
		                            col_start))
		{
			if (curr_state != IN_ROWS_SECTION) {
				std::cerr << "ERROR: at line " << line_num << std::endl
				          << next_line << std::endl;
				exit(EXIT_FAILURE);
			}

			curr_state = IN_COLUMNS_SECTION;
		}

		// Check if it is the start of the RHS section.
		else if (boost::regex_match(next_line.c_str(),
		                            matches,
		                            rhs_start))
		{
			if (curr_state != IN_COLUMNS_SECTION) {
				std::cerr << "ERROR: at line " << line_num << std::endl
				          << next_line << std::endl;
				exit(EXIT_FAILURE);
			}

			curr_state = IN_RHS_SECTION;
		}

		// Check if it is a column line.
		else if (boost::regex_match(next_line.c_str(),
		                            matches,
		                            col_line)
		         && curr_state != IN_RHS_SECTION)
		{
			if (curr_state != IN_COLUMNS_SECTION) {
				std::cerr << "ERROR: at line " << line_num << std::endl
				          << next_line << std::endl;
				exit(EXIT_FAILURE);
			}

			handle_col_line(matches, problem_instance);
		}

		// Check it is a RHS line.
		else if (boost::regex_match(next_line.c_str(),
		                            matches,
		                            rhs_line))
		{
			if (curr_state != IN_RHS_SECTION) {
				std::cerr << "ERROR: at line " << line_num << std::endl
				          << next_line << std::endl;
				exit(EXIT_FAILURE);
			}

			handle_rhs_line(matches, problem_instance);
		}

		// Check if it is the start of the BOUNDS section.
		else if (boost::regex_match(next_line.c_str(),
		                            matches,
		                            bounds_start))
		{
			if (curr_state != IN_RHS_SECTION) {
				std::cerr << "ERROR: at line " << line_num << std::endl
				          << next_line << std::endl;
				exit(EXIT_FAILURE);
			}

			curr_state = IN_BOUNDS_SECTION;
		}

		// Check if it is in the BOUNDS line
		else if (boost::regex_match(next_line.c_str(),
									matches,
									bounds_line))
		{
			if (curr_state != IN_BOUNDS_SECTION) {
				std::cerr << "ERROR: at line " << line_num << std::endl
				          << next_line << std::endl;
				exit(EXIT_FAILURE);
			}

			handle_bounds_line(matches, problem_instance, line_num);
		}

		// This is an unrecognized line.  It might be part of the MPS
		// format that still hasn't been implemented.
		else {
			std::cerr << "ERROR: Unrecognized line type at line "
			          << line_num << std::endl
			          << next_line
			          << std::endl;
			exit(EXIT_FAILURE);
		}

		line_num++;
	}

	return problem_instance;
}
