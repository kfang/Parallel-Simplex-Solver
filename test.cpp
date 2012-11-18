#include <iostream>
#include <stdlib.h>
#include <cmath>
/*
 * This calls TableauTest.
 */
using namespace std;

int main(){

	TableauTest test1;
	int rows;
	int cols;
	int* current_basis;
	int* candidates;

	test1=new TableauTest(true);
	test1.printMat();
}


