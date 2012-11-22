////////////////////////////////////////////////////////////////////////////
//                           simplex_problem.h                            //
////////////////////////////////////////////////////////////////////////////

#ifndef SIMPLEX_PROBLEM_H
#define SIMPLEX_PROBLEM_H

#include <string>
#include <map>
#include <vector>

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
	constraint_type get_type(void);
	void set_coefficient(const std::string& name, float value);
	float get_coefficient(const std::string& name);
	void set_rhs(float value);
	std::string get_string_representation(void);

private:
};

//--------------------------------------------------------------------------
// OBJECTIVE_FUNCTION

enum optimization_type { MAX, MIN };

class Objective_Function
{
public:
	// Constructors and destructor.
	Objective_Function(void);
	~Objective_Function(void);

	// methods
	optimization_type get_type(void);
	void set_coefficient(const std::string& name, float value);
	float get_coefficient(const std::string& name);
	std::string get_string_representation(void);

private:
};

//--------------------------------------------------------------------------
// SIMPLEX_PROBLEM

class Simplex_Problem
{
public:
	// Constructors and destructor.
	Simplex_Problem(void);
	Simplex_Problem(const std::string name);
	~Simplex_Problem(void);

	// methods
	void set_name(const std::string& name);
	void add_constraint(const std::string& name, constraint_type type);
	void add_obj_func(const std::string& name, optimization_type type);
	void set_constraint_coeff(const std::string& constraint_name,
	                          const std::string& coeff_name,
	                          const float& coeff_value);
	void set_constraint_rhs(const std::string& constraint_name,
	                        const float& coeff_value);
	void print(void);

private:
	std::string name;
	std::vector<std::string> variables;
	std::vector<std::string> constraint_names;
	std::map<std::string, Constraint> constraints;
	Objective_Function obj_func;
};

#endif
