#include <iostream>
#include <string>
#include <map>
#include <list>
#include <stdlib.h>
#include <vector>
#include "GridPos.cpp"
#include "Tableau.cpp"

using namespace std;

class Solver {
	Tableau tableau;
public:
	Solver (Tableau);
	void solve();
	vector<int> possible_incoming();
	int select_incoming(vector<int>);
	int select_outgoing();
	bool isUnbounded();
	void pivot(int, int);
	bool check_if_done();
};

void Solver::solve() {
	while(!check_if_done()) {
		int incoming = select_incoming(possible_incoming());
		int outgoing = select_outgoing();
		if (isUnbounded()) {
			break;
		}
		pivot(incoming, outgoing);
	}
	// Don't actually know what we are going to return here
}

void Solver::pivot(int incoming, int outgoing) {
	tableau.pivot(incoming, outgoing);
}

//might want to use blas
vector<int> Solver::possible_incoming(){

}

int Solver::select_incoming(vector<int> possible_incoming){
	return possible_incoming[0];
}


int Solver::select_outgoing(){

}


//if all negatives in a column, infinite solutions, return true
bool Solver::isUnbounded(){

}


//check if top row of tableau is all positive
bool Solver::check_if_done(){

}
