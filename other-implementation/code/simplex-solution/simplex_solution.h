////////////////////////////////////////////////////////////////////////////
//                           simplex_solution.h                           //
////////////////////////////////////////////////////////////////////////////

#ifndef SIMPLEX_SOLUTION_H
#define SIMPLEX_SOLUTION_H

#include <string>

//--------------------------------------------------------------------------
// SIMPLEX_SOLUTION

enum solution_type { OPTIMAL_SOLUTION_FOUND,
                     UNBOUNDED_SOLUTION,
                     INFEASIBLE_PROBLEM };

class Simplex_Solution
{
public:
	// Constructors and destructor.
	Simplex_Solution(void);
	Simplex_Solution(const std::string& name,
	                 solution_type type,
	                 double run_time,
	                 int num_iterations);
	~Simplex_Solution(void);

	// methods
	std::string get_name(void);
	void set_name(const std::string& name);
	solution_type get_type(void);
	void set_type(const solution_type& type);
	int get_num_variables(void);
	void set_num_variables(const int& num_variables);
	int get_num_constraints(void);
	void set_num_constraints(const int& num_constraints);
	double get_run_time(void);
	void set_run_time(const double& run_time);
	int get_num_iterations(void);
	void set_num_iterations(const int& num_iterations);
	float get_obj_func_val(void);
	void set_obj_func_val(const float& obj_func_val);
	void increment_num_iterations(void);
	std::string get_solution_info();

private:
	std::string name;
	solution_type type;
	int num_variables;
	int num_constraints;
	double run_time;
	int num_iterations;
	float obj_func_val;
};

#endif
