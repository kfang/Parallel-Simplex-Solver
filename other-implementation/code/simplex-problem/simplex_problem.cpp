////////////////////////////////////////////////////////////////////////////
//                          simplex_problem.cpp                           //
////////////////////////////////////////////////////////////////////////////

#include "simplex_problem.h"

/////////////////////////////// CONSTRAINT /////////////////////////////////

//--------------------------------------------------------------------------
// CONSTRUCTOR AND DESTRUCTOR

Constraint::Constraint(const std::string& name, constraint_type type)
{
	// TODO
}

Constraint::~Constraint(void)
{
	// TODO
}

//--------------------------------------------------------------------------
// GET_TYPE

constraint_type Constraint::get_type(void)
{
	// TODO
}

//--------------------------------------------------------------------------
// SET_COEFFICIENT

void Constraint::set_coefficient(const std::string& name, float value)
{
	// TODO
}

//--------------------------------------------------------------------------
// GET_COEFFICIENT

float Constraint::get_coefficient(const std::string& name)
{
	// TODO
}

//--------------------------------------------------------------------------
// SET_RHS

void Constraint::set_rhs(float value)
{
	// TODO
}

//--------------------------------------------------------------------------
// GET_STRING_REPRESENTATION

std::string Constraint::get_string_representation(void)
{
	// TODO
}


/////////////////////////// OBJECTIVE_FUNCTION /////////////////////////////

Objective_Function::Objective_Function(void)
{
	// TODO
}

Objective_Function::~Objective_Function(void)
{
	// TODO
}

//--------------------------------------------------------------------------
// GET TYPE

optimization_type Objective_Function::get_type(void)
{
	// TODO
}

//--------------------------------------------------------------------------
// SET COEFFICIENT

void Objective_Function::set_coefficient(const std::string& name, float value)
{
	// TODO
}

//--------------------------------------------------------------------------
// GET_COEFFICIENT

float Objective_Function::get_coefficient(const std::string& name)
{
	// TODO
}

//--------------------------------------------------------------------------
// STRING_REPRESENTATION

std::string get_string_representation(void)
{
	// TODO
}

///////////////////////////// SIMPLEX PROBLEM //////////////////////////////

//--------------------------------------------------------------------------
// CONSTRUCTOR AND DESTRUCTOR

Simplex_Problem::Simplex_Problem(void)
{
	// TODO
}

Simplex_Problem::~Simplex_Problem(void)
{
	// TODO
}

//--------------------------------------------------------------------------
// SET_NAME

void Simplex_Problem::set_name(const std::string& name)
{
	// TODO
}

//--------------------------------------------------------------------------
// ADD_CONSTRAINT

void Simplex_Problem::add_constraint(const std::string& name, constraint_type type)
{
	// TODO
}

//--------------------------------------------------------------------------
// ADD_OBJ_FUNC

void Simplex_Problem::add_obj_func(const std::string& name, optimization_type type)
{
	// TODO
}

//--------------------------------------------------------------------------
// SET_CONSTRAINT_COEFF

void Simplex_Problem::set_constraint_coeff(const std::string& constraint_name,
                                           const std::string& coeff_name,
                                           const float& coeff_value)
{
	// TODO
}

//--------------------------------------------------------------------------
// SET_CONSTRAINT_RHS

void Simplex_Problem::set_constraint_rhs(const std::string& constraint_name,
                                         const float& coeff_value)
{
	// TODO
}

//--------------------------------------------------------------------------
// PRINT

void Simplex_Problem::print(void)
{
	// TODO
}
