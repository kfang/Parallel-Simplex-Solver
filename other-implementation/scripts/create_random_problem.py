#!/usr/bin/env python2
############################################################################
#                        create_random_problem.py                          #
############################################################################

import sys
import optparse
import random

#---------------------------------------------------------------------------
# SIMPLEXPROBLEM

class SimplexProblem(object):
	def __init__(self, name):
		self.name = name
		self.variable_names = []
		self.constraint_names = []
		self.constraints = {}
		self.obj_func = {}

	def add_variable_name(self, name):
		self.variable_names.append(name)

	def add_constraint_name(self, name):
		self.constraint_names.append(name)
		self.constraints[name] = {}

	def set_constraint_coeff(self, constraint_name, var_name, coeff_val):
		if constraint_name not in self.constraints:
			self.constraints[constraint_name] = {}
		self.constraints[constraint_name][var_name] = coeff_val

	def set_constraint_rhs(self, constraint_name, rhs_val):
		if constraint_name not in self.constraints:
			self.constraints[constraint_name] = {}
		self.constraints[constraint_name]["rhs"] = rhs_val

	def set_obj_func_coeff(self, var_name, coeff_val):
		self.obj_func[var_name] = coeff_val

	def __str__(self):
		# The final mps representation of the problem will be in this varialbe.
		mps_string = ""

		# NAME section
		mps_string += "NAME {0}\n\n".format(self.name)

		# ROWS section
		mps_string += "ROWS\n"
		mps_string += "N    {0}\n".format('COST')
		for constraint_name in self.constraint_names:
			mps_string += "L    {0}\n".format(constraint_name)
		mps_string += "\n"

		# COLUMNS section
		mps_string += "COLUMNS\n"
		for variable_name in self.variable_names:
			# Add the a variable's value in the objective function.
			if variable_name in self.obj_func:
				obj_func_coeff = self.obj_func[variable_name]
				mps_string += "{0}    {1}    {2}\n".format(variable_name,
				                                           "COST",
				                                           obj_func_coeff)

			# Add the variable's value for each constraint.
			for constraint_name in self.constraint_names:
				if variable_name in self.constraints[constraint_name]:
					variable_coeff = self.constraints[constraint_name][variable_name]
					mps_string += "{0}    {1}    {2}\n".format(variable_name,
					                                           constraint_name,
					                                           variable_coeff)
		mps_string += "\n"

		# RHS section
		mps_string += "RHS\n"
		for constraint_name in self.constraint_names:
			rhs_val = self.constraints[constraint_name]["rhs"]
			mps_string += "B    {0}    {1}\n".format(constraint_name,
			                                         rhs_val)

		# ENDATA
		mps_string += "ENDATA"

		# Return the final mps representation.
		return mps_string

#---------------------------------------------------------------------------
# CREATE_RANDOM_PROBLEM

def create_random_problem(problem_name,
                          num_variables,
                          num_constraints,
                          variable_inclusion_probability,
                          upper_bound):
	problem = SimplexProblem(problem_name)

	# Add the variable names.
	for variable_id in range(num_variables):
		problem.add_variable_name('X' + str(variable_id))

	# Add the constraint names.
	for constraint_id in range(num_constraints):
		problem.add_constraint_name('C' + str(constraint_id))

	# Generate some random constraints.
	for constraint_name in problem.constraint_names:
		for variable_name in problem.variable_names:
			# Set the coefficients for some of the variables.
			if random.random() < variable_inclusion_probability:
				coeff_val = random.random()*upper_bound
				problem.set_constraint_coeff(constraint_name,
				                             variable_name,
				                             coeff_val)
		# Set the RHS for the constraint.
		rhs_val = random.random()*upper_bound
		problem.set_constraint_rhs(constraint_name, rhs_val)

	# Generate a random objective function.
	for variable_name in problem.variable_names:
		# Set the coefficients for some of the variables.
		if random.random() < variable_inclusion_probability:
			coeff_val = random.random()*upper_bound
			problem.set_obj_func_coeff(variable_name,
			                           coeff_val)

	# Return the finished problem.
	return problem

#---------------------------------------------------------------------------
# CREATE_CMDLINE_PARSER

def create_cmdline_parser():
	usage = "./create_random_problem.py [options]"
	desc = "Generate a random linear programming problem of given size." + \
	       "  The problem is outputted in the MPS file format."
	cmdline_parser = optparse.OptionParser(usage=usage,
	                                       description=desc)

	# Add options to the command line parser
	# size option
	cmdline_parser.add_option("-s", "--size", action="store", type="string",
	                          dest="size", metavar="SIZE",
	                          help="specify the number of variables and constraints in the linear programming problem")

	# name option
	cmdline_parser.add_option("--name", action="store", type="string",
	                          dest="name", metavar="NAME",
	                          help="specify the name given to the problem")

	# name option
	cmdline_parser.add_option("-p", "--probability", action="store", type="string",
	                          dest="probability", metavar="NUM",
	                          help="the probability a given variable will be included in a constraint")

	# name option
	cmdline_parser.add_option("-u", "--upper", action="store", type="string",
	                          dest="upper_bound", metavar="NUM",
	                          help="random numbers will range from [0 to upperbound)")

	return cmdline_parser

#---------------------------------------------------------------------------
# MAIN

if __name__ == '__main__':
	# Parse the command line.
	cmdline_parser = create_cmdline_parser()
	cmdline_options, _ = cmdline_parser.parse_args()

	assert cmdline_options.name and cmdline_options.size

	problem_name = cmdline_options.name
	problem_size = int(cmdline_options.size)

	variable_inclusion_probability = 0.3
	if cmdline_options.probability:
		variable_inclusion_probability = float(cmdline_options.probability)

	upper_bound = 50
	if cmdline_options.upper_bound:
		upper_bound = float(cmdline_options.upper_bound)

	problem = create_random_problem(problem_name,
	                                problem_size,
	                                problem_size,
	                                variable_inclusion_probability,
	                                upper_bound)

	# Print the problem to the command line.
	print problem
