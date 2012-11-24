#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <limits>

using namespace std;

class Tableau {
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
	Tableau (int, int);
	Tableau (bool);
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
	void update_candidate_cols(int* oldcands);
	int select_candidate_col();


};

Tableau::Tableau(int row, int col){
	rows = row;
	cols = col;
	matrix = new double*[rows];

	for (int i = 0; i < rows; i++){
		matrix[i] = new double[cols]();
	}
	current_basis = new int[rows-1];
}

Tableau::Tableau(bool example){
	if(example){

			rows = 4;
			cols = 8;
			matrix = new double*[rows];

			double rowset[][8]={{-5.0,-6.0,-9.0,-8.0, 0.0, 0.0, 0.0, 0.0},
			{ 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 6.0},
			{ 1.0, 2.0, 3.0, 1.0, 0.0, 1.0, 0.0, 5.0},
			{ 1.0, 1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 3.0}};

			int cb[3]={5,6,7};

			for (int i = 0; i < rows; i++){
				matrix[i] = new double[cols]();//();
			}

			current_basis = new int[rows-1];

			for(int i=0;i< rows-1;i++){
				current_basis[i]=cb[i];
			}

			for (int i = 0; i < rows; i++){
				for (int j = 0; j < cols; j++){
					matrix[i][j]= rowset[i][j];
				}
			}

			candidates=new int[cols-rows];
			int c[4]={0,0,0,0};
			int candSize=cols-rows;
			for(int i=0;i< candSize;i++){
				candidates[i]=c[i];
			}

		}
}

void Tableau::addRow(){
	//create a new matrix and copy the values over
	double** tempMat = new double*[rows + 1];
	for (int i = 0; i < rows; i++){
		tempMat[i] = new double[cols]();
		for(int j = 0; j < cols; j++){
			tempMat[i][j] = matrix[i][j];
		}
	}

	tempMat[rows] = new double[cols];
	for (int i = 0; i < cols; i++){
		tempMat[rows][cols] = 0.0;
	}

	rows++;
	delete[] matrix;
	matrix = tempMat;
}

void Tableau::addCol(){
	addCol(1);
}

void Tableau::addCol(int addNum){
	//iterate down the rows of the matrix, adding colums
	for (int i = 0; i < rows; i++){
		double *tempMat = new double[cols + addNum]();
		for (int j = 0; j < cols; j++){
			tempMat[j] = matrix[i][j];
		}
		delete[] matrix[i];
		matrix[i] = tempMat;
	}
	cols += addNum;
}

void Tableau::swapCols(int a, int b){
	for (int i = 0; i < rows; i++){
		int temp = matrix[i][a];
		matrix[i][a] = matrix[i][b];
		matrix[i][b] = temp;
	}
}

void Tableau::printMat(){
	//iterate over rows
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			cout << matrix[i][j];
			cout << " | ";
		}
		cout << '\n';
	}
}

void Tableau::pivot(int p, int q) {
	int j, k;
	// Calculate the values in the pth row.
	for (k = 0; k < cols; k++) {
		matrix[p][k] = matrix[p][k] / matrix[p][q];
	}
	// Calculate the values for all elements not in p or q
	for (j = 0; j < rows; j++) {
		for (k = 0; k < cols; k++) {
			if (j != p && k != q) {
				matrix[j][k] = matrix[j][k] - (matrix[p][k] * matrix[j][q]);
			}
		}
	}
	// Set the qth column to 0 also change the current basis
	for (j = 0; j < rows; j++) {
		matrix[j][q] = 0;
		if (current_basis[j] == q) {
			current_basis[j] = p;
		}
	}
	matrix[p][q] = 1;
}

int* Tableau::get_current_basis() {
	return current_basis;
}

//sequentially populate every cell
void Tableau::testPopulate(){
	double counter = 0.0;
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			matrix[i][j] = counter;
			counter ++;
		}
	}
}

// just a get function
int* Tableau::get_candidate_cols(){
	return candidates;
}

// Tested
/* Function: update_candidate_cols
 * Input: address to old candidates
 *
 * Explanation: This simply determines the columns within the non-base set
 */

void Tableau::update_candidate_cols(int* oldcands){

	int numVars=cols-1;

	int j=0;
	int k=0;

	// check if a col is in current_basis or not.
	for(int i=0;i<numVars; i++){
		if(*(current_basis+j)!=i){
			oldcands[k]=i;
			k++;
		}else{

			j++;
		}
	}

}

int Tableau::select_candidate_col(){

	double min=numeric_limits<double>::max( );
	int nextCol=0;

	for(int i=0;i<cols;i++){
		if(matrix[0][i]<min){
			min=matrix[0][i];
			nextCol=i;
		}
	}
	return nextCol;
}
