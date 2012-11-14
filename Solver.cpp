#include <iostream>
#include <string>
#include <map>
#include <list>
#include <stdlib.h>
#include <vector>
#include "GridPos.cpp"
#include "Tableu.cpp"

using namespace std;

class Solver {
	Tableau tableau;
public:
	Solver (Tableau);
	void solve();
	list<int> possible_incoming();
	int select_incoming(list<int>);
	int select_outgoing();
	bool unbounded_check();
	void pivot(int, int);
	bool check_if_done();
};

void Solver::solve() {
	while(!check_if_done()) {
		int incoming = select_incoming(possible_incoming());
		int outgoing = select_outgoing();
		if (unbounded_check()) {
			break;
		}
		pivot(incoming, outgoing);
	}
	// Don't actually know what we are going to return here
}
