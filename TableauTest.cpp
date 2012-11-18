#include <iostream>
#include <stdlib.h>
#include <cmath>

/*
 * This is just a hack attempt to unit test Tableau. It is select functions from Tableau.
 */

using namespace std;

class TableauTest {
	double** matrix;
	int rows;
	int cols;
	int* current_basis;
	int* candidates;

	//*****************************************************************************
	// these three are for getFeasibleIncoming. see function description for details
	int* feasibles[2];  // pointer to inFeasible and outFeasible pointers
	int* inFeasible;
	int* outFeasible;
	//*****************************************************************************

public:
	TableauTest (int, int);
	int height(){return rows;};
	int width(){return cols;};
	void printMat();
	void testPopulate();
	void addRow();
	void swapCols(int, int);
	void addCol();
	void addCol(int);
	double* operator [](int i){return matrix[i];};
	void pivot(int, int);
	int* get_current_basis();

	int* get_candidate_cols();
	bool getFeasibleIncoming(int* candidates);
	double* makeUnitVector(double* vector);
	bool isLinIndependent(double* vect1, double* vect2);
	bool makesNegativeSoln(double* sample_basis, double* b_vect);
	double* makeSampleBase(double* inCol, int outCol);
	double* getCol(int colNum);
	double* getB();

};

TableauTest::TableauTest(int row, int col){
	rows = row;
	cols = col;
	matrix = new double*[rows];

	for (int i = 0; i < rows; i++){
		matrix[i] = new double[cols]();
	}
	current_basis = new int[rows-1];
}

TableauTest::TableauTest(bool example){
	if(example){
		
		rows = 2;
		cols = 2;
		matrix = new double*[rows];

		for (int i = 0; i < rows; i++){
			matrix[i] = new double[cols]();
		}
		current_basis = new int[rows-1];
		
		matrix[0] = {1,3};
		matrix[1] = {2,4};
	}
	
}

void TableauTest::printMat(){
	//iterate over rows
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			cout << matrix[i][j];
			cout << " | ";
		}
		cout << '\n';
	}
}
