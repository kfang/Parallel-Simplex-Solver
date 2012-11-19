#include <iostream>
#include <stdlib.h>
#include <cmath>

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
	bool getFeasibleIncoming(int* candidates);
	void makeUnitVector(double* vector, double* unit);
	bool isLinIndependent(double* vect1, double* vect2);
	bool makesNegativeSoln(double* sample_basis, double* b_vect);
	double* makeSampleBase(double* inCol, int outCol);
	double* getCol(int colNum);
	double* getB();

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

// Tested
/* Function: makeUnitVector
 * Input: A vector to be made unit, a pointer to an empty vector
 * Output: no output, the unit vector gets filled with values
 *
 */
void Tableau::makeUnitVector(double* vector, double* unit){

	double norm=0;

	// get norm
	for(int i=0; i<cols-1;i++){
		double x=vector[i];
		double y=pow(x,2);
		norm=norm+y;
	}

	norm=sqrt(norm);

	for(int j=0; j<cols-1;j++){
		unit[j]=(vector[j]/norm);
	}

}

// Tested
/* Function: isLinIndepenedent
 * Input: two vectors of equal length (not necessarily normalized)
 * Output: boolean that indicates if the two vectors are linearly independent
 */
bool Tableau::isLinIndependent(double* vect1, double* vect2){
	int ln = cols-1;
	bool noMatch=false;

	double* uv1=new double[ln];
	double* uv2=new double[ln];

	for(int i=0;i< ln;i++){
			uv1[i]=0;
			uv2[i]=0;
	}

	makeUnitVector(vect1, uv1);
	makeUnitVector(vect2, uv2);

	//check for match
	for(int i=0;i<ln;i++){

		noMatch=noMatch||(uv1[i]!=uv2[i]);

	}

	free(uv1);
	free(uv2);
	return noMatch;
}


//TODO: This whole function: write and test
// b_vect is the last column of the matrix minus the first element
bool Tableau::makesNegativeSoln(double* sample_basis, double* b_vect){


	// define invBase:=get inverted form <- ACML
	// calculate matrix-vector product invBase*b_vect <- maybe ACML
	// return: does product have at least one negative element?

	return false;
}

//TODO: check for completeness and test
/* Function: getFeasibleIncoming
 * Input: candidates - pointer to a list of non-basis columns
 *        current_basis - pointer to the list of columns currently in the basis
 *        feasibles - pointer to an array of two int pointers
 *        	1. pointer[0] is new incoming set
 *          2. pointer[1] is new outgoing set
 * Output: a pointer to an updated array of two int pointers
 *
 * Explanation:
 * While the motivation behind this function is to produce a subset of incoming columns
 * in truth, the incoming and outgoing columns are coupled. This function populates two arrays,
 * inFeasible and outFeasible. The reasoning behind this can be found within what we need from
 * our incoming column. We want a column that is not linearly dependent with any of the columns
 * left within a previous base that has had a single column removed. This means that even if we
 * find such a linearly independent incoming column, it might not have been linearly independent
 * if we had left the outgoing column within the base. In other words the knowledge of our
 * incoming and outgoing columns is a joint property of the two columns. Chances are this will
 * not often be a problem but it can occur.
 *
 * NOTE: The result is that when we select an incoming column, we will also have a subset
 *       of outgoing choices from which to choose, possibly smaller than the base.
 *
 * NOTE2: This is definitely not the most efficient way to implement this
 */
int* Tableau::getFeasibleIncoming(int* candidates, int* feasibles){

	int ln = sizeof current_basis/sizeof current_basis[0];
	int lnc = sizeof candidates/sizeof cadidates[0];

	// define & malloc int* inFeasible
	int* inF;//=malloc();

	// define & malloc int* outFeasible
	int* outF;//=malloc();

	// for each outCol in current_basis
	for(int i=0;i<ln;i++){

		for(int k=0;k<lnc;k++){

			bool indep=true;

			// define sub_basis:= current_basis without outCol
			for(int j=0; j<ln; j++){
				if(*(current_basis+i)!=*(current_basis+j)){

					indep=indep&&isLinIndependent(getCol(i), getCol(j));
				}
			}

			if(indep){
				double* sample= makeSampleBase(candidate+j, i);
				if(!makesNegativeSoln(sample,getB(matrix))){
					inF+counter=candidates+j;
					outF+counter=current_basis+i;
				}
			}

		}
	}

	// ok what am I doing here? This isn't finished.
	//TODO: check the necessity and correctness of this approach
	inFeasible=inF;
	outFeasible=outF;
	return null;
}

// TODO: Test
// assumes column numbers begin with 0
double Tableau::getCol(int colNum){
	int ln = sizeof current_basis/sizeof current_basis[0];

	double col;
	for(int i=0;i<ln;i++){
		col+i=matrix[i][colNum];
	}
	return col;
}

double* Tableau::getB(){
	return (getCol(width()-1))+1;
}

//TODO: check this against algorithm for correctness
//TODO: check that cols and rows are correct
double* Tableau::makeSampleBase(double* inCol, int outCol){

	double* sample;//=malloc(sizeof(current_basis));
	int ln = sizeof current_basis/sizeof current_basis[0];

	for(int i=0;i<ln;i++){
		for(int j=0;j<width()-1;j++){

			if(i==outCol){
				sample[i][j]=inCol[j];
			}else{
				sample[i][j]=currentBasis[i][j];
			}

		}
	}
	return sample;
}
