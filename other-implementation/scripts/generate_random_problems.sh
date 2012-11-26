#!/usr/bin/env bash
############################################################################
#                       generate_random_problems.sh                        #
############################################################################

# Make a directory to put the generated problems in.
basedir=data/generated
mkdir -p $basedir

# The size variable is the number of variables and constraints to use in
# the randomly generated problem.
size_start=100
size_end=1000
size_step=100
sizes=`seq $size_start $size_step $size_end`

# This many random problems will be generated per size.
num_problems_per_size=15

# Generate the random problems and place them in the correct directory.
for size in $sizes;
do
	dirname=size-$size
	mkdir -p $basedir/$dirname
	echo "Generating $num_problems_per_size problems of size $size"
	for x in `seq 1 $num_problems_per_size`;
	do
		problem_name=size_"$size"_id_"$x"
		./scripts/create_random_problem.py --size=$size --name=$problem_name > $basedir/$dirname/$problem_name.mps;
	done
done
