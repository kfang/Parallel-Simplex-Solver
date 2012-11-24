////////////////////////////////////////////////////////////////////////////
//                           simplex_problem.h                            //
////////////////////////////////////////////////////////////////////////////

#ifndef SIMPLEX_PROBLEM_H
#define SIMPLEX_PROBLEM_H

#include <string>
#include <map>
#include <vector>
#include <set>

//--------------------------------------------------------------------------
// CONSTRAINT

enum constraint_type { LEQ, GEQ, EQ };

class Constraint
{
public:
	// Constructors and destructor.
	Constraint(const std::string& name, constraint_type type);
	~Constraint(void);

	// methods
	std::string get_name(void);
	constraint_type get_type(void);
	void set_coefficient(const std::string& name, float value);
	float get_coefficient(const std::string& name);
	void set_rhs(float value);
	float get_rhs(void);
	std::string get_string_representation(void);
	std::string get_string_representation(const std::vector<std::string>& variable_order);

private:
	std::string name;
	constraint_type type;
	std::map<std::string, float> constraint_coefficients;
	float rhs;
};

//--------------------------------------------------------------------------
// OBJECTIVE_FUNCTION

enum optimization_type { MAX, MIN };

class Objective_Function
{
public:
	// Constructors and destructor.
	Objective_Function(void);
	Objective_Function(const std::string& name, optimization_type type);
	~Objective_Function(void);

	// methods
	std::string get_name(void);
	void set_name(const std::string& name);
	optimization_type get_type(void);
	void set_type(optimization_type type);
	void set_coefficient(const std::string& coeff_name, float value);
	float get_coefficient(const std::string& coeff_name);
	std::string get_string_representation(void);
	std::string get_string_representation(const std::vector<std::string>& variable_order);

private:
	std::string name;
	optimization_type type;
	std::map<std::string, float> obj_func_coefficients;
};

//--------------------------------------------------------------------------
// SIMPLEX_PROBLEM

class Simplex_Problem
{
public:
	// Constructors and destructor.
	Simplex_Problem(void);
	Simplex_Problem(const std::string& name);
	~Simplex_Problem(void);

	// methods
	void set_name(const std::string& name);
	void add_constraint(const std::string& name, constraint_type type);
	void add_obj_func(const std::string& name, optimization_type type);
	void set_constraint_coeff(const std::string& constraint_name,
	                          const std::string& coeff_name,
	                          const float& coeff_value);
	float get_constraint_coeff(const std::string& constraint_name,
	                          const std::string& coeff_name);
	void set_constraint_rhs(const std::string& constraint_name,
	                        const float& rhs_value);
	float get_constraint_rhs(const std::string& constraint_name);
	void set_obj_coeff(const std::string& coeff_name,
	                   const float& coeff_value);
	float get_obj_coeff(const std::string& coeff_name);
	bool is_obj_func(const std::string& name);
	int get_num_variables(void);
	int get_num_constraints(void);
	Constraint& get_constraint(const std::string& name);
	void print(void);

	// Provide iterators to the variables and constraints.
	typedef std::vector<std::string>::iterator variable_iterator;
	typedef std::vector<std::string>::iterator constraint_name_iterator;

	variable_iterator get_variable_iterator(void);
	variable_iterator get_variable_end(void);
	constraint_name_iterator get_constraint_name_iterator(void);
	constraint_name_iterator get_constraint_name_end(void);

private:
	std::string name;
	std::set<std::string> variables;
	std::vector<std::string> variable_order;
	std::vector<std::string> constraint_names;
	std::map<std::string, Constraint*> constraints;
	Objective_Function obj_func;
};

#endif
