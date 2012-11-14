#include <iostream>
#include <stdlib.h>
#include <vector>

using namespace std;

class Solver {
	Tableau* tableau;
public:
	Solver (Tableau*);
	void solve();
	vector<int> possible_incoming();
	int select_incoming(vector<int>);
	int select_outgoing(int);
	bool isUnbounded();
	void pivot(int, int);
	bool check_if_done();
};

Solver::Solver(Tableau* t){
	tableau = t;
}

void Solver::solve() {
	while(!check_if_done()) {
		int incoming = select_incoming(possible_incoming());
		int outgoing = select_outgoing(incoming);
		if (isUnbounded()) {
			break;
		}
		pivot(incoming, outgoing);
	}
	// Don't actually know what we are going to return here
}

void Solver::pivot(int incoming, int outgoing) {
	tableau->pivot(incoming, outgoing);
}

//might want to use blas
vector<int> Solver::possible_incoming(){

}

int Solver::select_incoming(vector<int> possible_incoming){
	return possible_incoming[0];
}


int Solver::select_outgoing(int incoming){
	int* current_basis = tableau.get_current_basis;
	int rows = tableau.height() - 1;
	int cols = tableau.cols();
	double* row = tableau[current_basis[0]];
	double current_val = row[cols] / row[incoming];
	double min = current_val;
	int outgoing = current_basis[0];
	for (int i = 1; i < rows; i++) {
		row = tableau[current_basis[i]];
		current_val = row[cols] / row[incoming];
		if (current_val < min) {
			min = current_val;
			outgoing = current_basis[i];
		}
	}
	return outgoing;
}


//if all negatives in a column, infinite solutions, return true
bool Solver::isUnbounded(){
	//iterate through the columns and check each row to see if its negative
	for(int col = 0; col < tableau->width(); col++){
		//boolean to track if we've seen a positive number in a column
		bool negativeCol = true;

		//go through the rows and check if the number is positive
		for (int row = 0; row < tableau->height(); row++){
			if ((*tableau)[row][col] >= 0.0){
				//found a positive number, not a negative column, break out of for loop
				negativeCol = false;
				break;
			}
		}

		//if we've gone through the entire column and negativeCol is still true,
		//there were no positive numbers in the column, return true
		if (negativeCol){
			return true;
		}
	}

	//went through all the columns, none were all negative
	return false;

}


//check if top row of tableau is all positive
bool Solver::check_if_done(){
	//iterate through the top row
	for (int col = 0; col < tableau->width(); col++){
		//if the number is negative, return false
		if ((*tableau)[0][col] < 0.0){
			return false;
		}
	}

	//didn't find any negative numbers, therefore they're all
	//positive
	return true;
}
