////////////////////////////////////////////////////////////////////////////
//                          simplex_problem.cpp                           //
////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <sstream>
#include <cmath>
#include <set>
#include "simplex_problem.h"

#define ABS(X) (((X) < 0) ? (-1*(X)) : (X))

/////////////////////////////// CONSTRAINT /////////////////////////////////

//--------------------------------------------------------------------------
// CONSTRUCTOR AND DESTRUCTOR

Constraint::Constraint(const std::string& name, constraint_type type)
: constraint_coefficients()
{
	this->name = name;
	this->type = type;
}

Constraint::~Constraint(void)
{
}

//--------------------------------------------------------------------------
// GET_NAME

std::string Constraint::get_name(void)
{
	return name;
}

//--------------------------------------------------------------------------
// GET_TYPE

constraint_type Constraint::get_type(void)
{
	return type;
}

//--------------------------------------------------------------------------
// SET_COEFFICIENT

void Constraint::set_coefficient(const std::string& name, float value)
{
	constraint_coefficients[name] = value;
}

//--------------------------------------------------------------------------
// GET_COEFFICIENT

float Constraint::get_coefficient(const std::string& name)
{
	// The value of the coefficient is whatever it was set to or 0 if it
	// was never set.
	float val = constraint_coefficients.find(name) != constraint_coefficients.end() ?
	            constraint_coefficients[name] :
	            0;
	return val;
}

//--------------------------------------------------------------------------
// SET_RHS

void Constraint::set_rhs(float value)
{
	rhs = value;
}

//--------------------------------------------------------------------------
// GET_RHS

float Constraint::get_rhs(void)
{
	return rhs;
}

//--------------------------------------------------------------------------
// GET_STRING_REPRESENTATION

std::string Constraint::get_string_representation(void)
{
	std::stringstream representation;
	bool first_variable = true;
	for (std::map<std::string, float>::iterator iter = constraint_coefficients.begin();
	     iter != constraint_coefficients.end();
	     iter++)
	{
		const std::string& variable = iter->first;
		float val = constraint_coefficients[variable];

		if (val != 0) {
			// Add the sign.
			bool is_negative = val < 0;

			// This is the first variable, add the sign without spaces.  Don't
			// print leading +'s
			if (first_variable) {
				if (is_negative) {
					representation << "-";
				}
				first_variable = false;
			} else {
				if (is_negative) {
					representation << " - ";
				} else {
					representation << " + ";
				}
			}

			// Add the coefficient and variable name.
			if (ABS(val) != 1)
				representation << ABS(val) << "*";
			representation << variable;
		}
	}

	// Add the constraint.
	// Less than equal to constraint.
	if (type == LEQ) {
		representation << " <= ";
	}

	// Less than equal to constraint.
	else if (type == GEQ) {
		representation << " >= ";
	}

	// Less than equal to constraint.
	else {
		representation << " = ";
	}

	// Add the RHS
	representation << rhs;

	return representation.str();
}


std::string Constraint::get_string_representation(const std::vector<std::string>& variable_order)
{
	std::stringstream representation;

	// Add the variables.
	bool first_variable = true;
	for (int i = 0; i < variable_order.size(); i++)
	{
		const std::string& variable = variable_order[i];
		float val = constraint_coefficients[variable];

		if (val != 0) {
			// Add the sign.
			bool is_negative = val < 0;

			// This is the first variable, add the sign without spaces.  Don't
			// print leading +'s
			if (first_variable) {
				if (is_negative) {
					representation << "-";
				}
				first_variable = false;
			} else {
				if (is_negative) {
					representation << " - ";
				} else {
					representation << " + ";
				}
			}

			// Add the coefficient and variable name.
			if (ABS(val) != 1)
				representation << ABS(val) << "*";
			representation << variable;
		}
	}

	// Add the constraint.
	// Less than equal to constraint.
	if (type == LEQ) {
		representation << " <= ";
	}

	// Less than equal to constraint.
	else if (type == GEQ) {
		representation << " >= ";
	}

	// Less than equal to constraint.
	else {
		representation << " = ";
	}

	// Add the RHS
	representation << rhs;

	return representation.str();
}


/////////////////////////// OBJECTIVE_FUNCTION /////////////////////////////

//--------------------------------------------------------------------------
// CONSTRUCTOR AND DESTRUCTOR

Objective_Function::Objective_Function(void)
{
}

Objective_Function::Objective_Function(const std::string& name, optimization_type type)
{
	this->name = name;
	this->type = type;
}

Objective_Function::~Objective_Function(void)
{
}

//--------------------------------------------------------------------------
// GET_NAME

std::string Objective_Function::get_name(void)
{
	return name;
}

//--------------------------------------------------------------------------
// SET_NAME

void Objective_Function::set_name(const std::string& name)
{
	this->name = name;
}


//--------------------------------------------------------------------------
// GET_TYPE

optimization_type Objective_Function::get_type(void)
{
	return type;
}

//--------------------------------------------------------------------------
// SET_TYPE

void Objective_Function::set_type(optimization_type type)
{
	this->type = type;
}

//--------------------------------------------------------------------------
// SET COEFFICIENT

void Objective_Function::set_coefficient(const std::string& name, float value)
{
	obj_func_coefficients[name] = value;
}

//--------------------------------------------------------------------------
// GET_COEFFICIENT

float Objective_Function::get_coefficient(const std::string& name)
{
	// The value of the coefficient is whatever it was set to or 0 if it
	// was never set.
	float val = obj_func_coefficients.find(name) != obj_func_coefficients.end() ?
	            obj_func_coefficients[name] :
	            0;
	return val;
}

//--------------------------------------------------------------------------
// STRING_REPRESENTATION

std::string Objective_Function::get_string_representation(void)
{
	std::stringstream representation;

	if (type == MAX) {
		representation << "MAX ";
	} else {
		representation << "MIN ";
	}

	bool first_variable = true;
	for (std::map<std::string, float>::iterator iter = obj_func_coefficients.begin();
	     iter != obj_func_coefficients.end();
	     iter++)
	{
		const std::string& variable = iter->first;
		float val = obj_func_coefficients[variable];

		if (val != 0) {
			// Add the sign.
			bool is_negative = val < 0;

			// This is the first variable, add the sign without spaces.  Don't
			// print leading +'s
			if (first_variable) {
				if (is_negative) {
					representation << " -";
				}
				first_variable = false;
			} else {
				if (is_negative) {
					representation << " - ";
				} else {
					representation << " + ";
				}
			}

			// Add the coefficient and variable name.
			if (ABS(val) != 1)
				representation << ABS(val) << "*";
			representation << variable;
		}
	}

	return representation.str();
}

std::string Objective_Function::get_string_representation(const std::vector<std::string>& variable_order)
{
	std::stringstream representation;

	if (type == MAX) {
		representation << "MAX ";
	} else {
		representation << "MIN ";
	}

	bool first_variable = true;
	for (int i = 0; i < variable_order.size(); i++)
	{
		const std::string& variable = variable_order[i];
		float val = obj_func_coefficients[variable];

		if (val != 0) {
			// Add the sign.
			bool is_negative = val < 0;

			// This is the first variable, add the sign without spaces.  Don't
			// print leading +'s
			if (first_variable) {
				if (is_negative) {
					representation << " -";
				}
				first_variable = false;
			} else {
				if (is_negative) {
					representation << " - ";
				} else {
					representation << " + ";
				}
			}

			// Add the coefficient and variable name.
			if (ABS(val) != 1)
				representation << ABS(val) << "*";
			representation << variable;
		}
	}

	return representation.str();
}

///////////////////////////// SIMPLEX PROBLEM //////////////////////////////

//--------------------------------------------------------------------------
// CONSTRUCTORS AND DESTRUCTOR

Simplex_Problem::Simplex_Problem(void)
{
}

Simplex_Problem::Simplex_Problem(const std::string& name)
{
	this->name = name;
}

Simplex_Problem::~Simplex_Problem(void)
{
	for (std::map<std::string, Constraint*>::iterator iter = constraints.begin();
	     iter != constraints.end();
	     iter++)
	{
		const std::string& constraint_name = iter->first;
		delete constraints[constraint_name];
	}
}

//--------------------------------------------------------------------------
// SET_NAME

void Simplex_Problem::set_name(const std::string& name)
{
	this->name = name;
}

//--------------------------------------------------------------------------
// GET_NAME

std::string Simplex_Problem::get_name(void)
{
	return name;
}

//--------------------------------------------------------------------------
// ADD_CONSTRAINT

void Simplex_Problem::add_constraint(const std::string& name, constraint_type type)
{
	constraint_names.push_back(name);
	constraints[name] = new Constraint(name, type);
}

//--------------------------------------------------------------------------
// ADD_OBJ_FUNC

void Simplex_Problem::add_obj_func(const std::string& name, optimization_type type)
{
	obj_func = Objective_Function(name, type);
}

//--------------------------------------------------------------------------
// SET_CONSTRAINT_COEFF

void Simplex_Problem::set_constraint_coeff(const std::string& constraint_name,
                                           const std::string& coeff_name,
                                           const float& coeff_value)
{
	bool variable_already_added = variables.find(coeff_name) != variables.end();
	if (!variable_already_added) {
		variables.insert(coeff_name);
		variable_order.push_back(coeff_name);
	}
	constraints[constraint_name]->set_coefficient(coeff_name, coeff_value);
}

//--------------------------------------------------------------------------
// GET_CONSTRAINT_COEFF

float Simplex_Problem::get_constraint_coeff(const std::string& constraint_name,
                                           const std::string& coeff_name)
{
	return constraints[constraint_name]->get_coefficient(coeff_name);
}

//--------------------------------------------------------------------------
// SET_CONSTRAINT_RHS

void Simplex_Problem::set_constraint_rhs(const std::string& constraint_name,
                                         const float& rhs_value)
{
	constraints[constraint_name]->set_rhs(rhs_value);
}

//--------------------------------------------------------------------------
// GET_CONSTRAINT_RHS

float Simplex_Problem::get_constraint_rhs(const std::string& constraint_name)
{
	return constraints[constraint_name]->get_rhs();
}

//--------------------------------------------------------------------------
// SET_OBJ_COEFF

void Simplex_Problem::set_obj_coeff(const std::string& coeff_name,
                                    const float& coeff_value)
{
	obj_func.set_coefficient(coeff_name, coeff_value);
}

//--------------------------------------------------------------------------
// GET_OBJ_COEFF

float Simplex_Problem::get_obj_coeff(const std::string& coeff_name)
{
	return obj_func.get_coefficient(coeff_name);
}

//--------------------------------------------------------------------------
// IS_OBJ_FUNC

bool Simplex_Problem::is_obj_func(const std::string& name)
{
	return obj_func.get_name() == name;
}

//--------------------------------------------------------------------------
// GET_NUM_VARIABLES

int Simplex_Problem::get_num_variables(void)
{
	return variables.size();
}

//--------------------------------------------------------------------------
// GET_NUM_CONSTRAINTS

int Simplex_Problem::get_num_constraints(void)
{
	return constraint_names.size();
}

//--------------------------------------------------------------------------
// GET_CONSTRAINT

Constraint& Simplex_Problem::get_constraint(const std::string& name)
{
	return *constraints[name];
}

//--------------------------------------------------------------------------
// GET_VARIABLE_ITERATOR

Simplex_Problem::variable_iterator Simplex_Problem::get_variable_iterator(void)
{
	return variable_order.begin();
}

//--------------------------------------------------------------------------
// GET_VARIABLE_END

Simplex_Problem::variable_iterator Simplex_Problem::get_variable_end(void)
{
	return variable_order.end();
}

//--------------------------------------------------------------------------
// GET_CONSTRAINT_NAME_ITERATOR

Simplex_Problem::constraint_name_iterator Simplex_Problem::get_constraint_name_iterator(void)
{
	return constraint_names.begin();
}

//--------------------------------------------------------------------------
// GET_CONSTRAINT_NAME_END

Simplex_Problem::constraint_name_iterator Simplex_Problem::get_constraint_name_end(void)
{
	return constraint_names.end();
}

//--------------------------------------------------------------------------
// PRINT

void Simplex_Problem::print(void)
{
	std::cout << name << std::endl;
	std::cout << obj_func.get_string_representation(variable_order) << std::endl;

	for (std::map<std::string, Constraint*>::iterator iter = constraints.begin();
	     iter != constraints.end();
	     iter++)
	{
		const std::string& constraint_name = iter->first;
		Constraint* constraint = constraints[constraint_name];
		std::cout << constraint->get_string_representation(variable_order) << std::endl;
	}
}
