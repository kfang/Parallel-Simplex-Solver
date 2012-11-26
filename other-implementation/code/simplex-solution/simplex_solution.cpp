////////////////////////////////////////////////////////////////////////////
//                          simplex_solution.cpp                          //
////////////////////////////////////////////////////////////////////////////

#include <string>
#include <sstream>
#include "simplex_solution.h"

//--------------------------------------------------------------------------
// CONSTRUCTOR AND DESTRUCTOR

Simplex_Solution::Simplex_Solution(void)
{
	num_iterations = 0;
}

Simplex_Solution::Simplex_Solution(const std::string& name,
                                   solution_type type,
                                   double run_time,
                                   int num_iterations)
{
	this->name = name;
	this->type = type;
	this->run_time = run_time;
	this->num_iterations = num_iterations;
}

Simplex_Solution::~Simplex_Solution(void)
{
}

//--------------------------------------------------------------------------
// GET_NAME

std::string Simplex_Solution::get_name(void)
{
	return name;
}

//--------------------------------------------------------------------------
// SET_NAME

void Simplex_Solution::set_name(const std::string& name)
{
	this->name = name;
}

//--------------------------------------------------------------------------
// GET_TYPE

solution_type Simplex_Solution::get_type(void)
{
	return type;
}

//--------------------------------------------------------------------------
// SET_TYPE

void Simplex_Solution::set_type(const solution_type& type)
{
	this->type = type;
}

//--------------------------------------------------------------------------
// GET_NUM_VARIABLES

int Simplex_Solution::get_num_variables(void)
{
	return num_variables;
}

//--------------------------------------------------------------------------
// SET_NUM_VARIABLES

void Simplex_Solution::set_num_variables(const int& num_variables)
{
	this->num_variables = num_variables;
}

//--------------------------------------------------------------------------
// GET_NUM_CONSTRAINTS

int Simplex_Solution::get_num_constraints(void)
{
	return num_constraints;
}

//--------------------------------------------------------------------------
// SET_NUM_CONSTRAINTS

void Simplex_Solution::set_num_constraints(const int& num_constraints)
{
	this->num_constraints = num_constraints;
}

//--------------------------------------------------------------------------
// GET_RUN_TIME

double Simplex_Solution::get_run_time(void)
{
	return run_time;
}

//--------------------------------------------------------------------------
// SET_RUN_TIME

void Simplex_Solution::set_run_time(const double& run_time)
{
	this->run_time = run_time;
}

//--------------------------------------------------------------------------
// GET_NUM_ITERATIONS

int Simplex_Solution::get_num_iterations(void)
{
	return num_iterations;
}

//--------------------------------------------------------------------------
// SET_NUM_ITERATIONS

void Simplex_Solution::set_num_iterations(const int& num_iterations)
{
	this->num_iterations = num_iterations;
}

//--------------------------------------------------------------------------
// GET_OBJ_FUNC_VAL

float Simplex_Solution::get_obj_func_val(void)
{
	return obj_func_val;
}

//--------------------------------------------------------------------------
// SET_OBJ_FUNC_VAL

void Simplex_Solution::set_obj_func_val(const float& obj_func_val)
{
	this->obj_func_val = obj_func_val;
}

//--------------------------------------------------------------------------
// INCREMENT_NUM_ITERATIONS

void Simplex_Solution::increment_num_iterations(void)
{
	num_iterations++;
}

//--------------------------------------------------------------------------
// GET_STATISTICS

std::string Simplex_Solution::get_solution_info(void)
{
	std::stringstream info;
	info << "NAME:            " << name << "\n";
	info << "TYPE:            ";
	switch (type) {
		case OPTIMAL_SOLUTION_FOUND:
			info << "OPTIMAL_SOLUTION_FOUND\n";
			break;
		case UNBOUNDED_SOLUTION:
			info << "UNBOUNDED_SOLUTION\n";
			break;
		case INFEASIBLE_PROBLEM:
			info << "INFEASIBLE_PROBLEM\n";
			break;
	}
	info << "NUM VARIABLES:   " << num_variables << "\n";
	info << "NUM CONSTRAINTS: " << num_constraints << "\n";
	info << "RUN TIME:        " << run_time << " seconds" << "\n";
	info << "OBJ FUNC VAL:    " << obj_func_val << "\n";
	info << "NUM ITERATIONS:  " << num_iterations;

	return info.str();
}
