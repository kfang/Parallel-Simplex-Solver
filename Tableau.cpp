#include <iostream>
using namespace std;

class Tableau {
	double** matrix;
	int rows;
	int cols;
	int* current_basis;

	//*****************************************************************************
	// these three are for getFeasibleIncoming. see function description for details
	int* feasibles[2];  // pointer to inFeasible and outFeasible pointers
	int* inFeasible;
	int* outFeasible;
	//*****************************************************************************

public:
	Tableau (int, int);
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

	int* get_candidate_cols(int *current_basis, int numVars);
	bool getFeasibleIncoming(int* candidates, int* current_basis);
	double* makeUnitVector(double* vector);
	bool isLinIndependent(double* vect1, double* vect2);
	bool makesNegativeSoln(double* sample_basis, double* b_vect);
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

/* Function: get_candidate_cols
 * Input: int *current_basis - the pointer to the current basis
 *        int numVars - the number of total columns
 *
 * Explanation: This simply determines the columns within the non-base set
 */

int* Tableau::get_candidate_cols(){

	numVars=width()-1;

	// how many variables aren't already within the current basis?
	int *candidates = (int*) malloc(numVars*sizeof(int)-sizeof(*current_basis));

	int j=0;
	int k=0;

	// check if a col is in current_basis or not.
	for(int i=0;i<numVars; i++){
		if((*current_basis+j)!=i){
			*candidates+k=i;
			k++;
		}else{
			j++;
		}
	}
	return candidates;

}


/* Function: makeUnitVector
 * Input: A vector
 * Output: A unit vector based upon the input vector
 */
double* Tableau::makeUnitVector(double* vector){

	double norm=0;
	int ln = sizeof vector/sizeof vector[0];

	// get norm
	for(int i=0; i<ln;i++){
		double x=*(vector+i);
		norm=norm+pow(x,2);
	}

	norm=pow(norm,(1/2));
	//make space for unit array
	double *unit = malloc(ln*sizeof(double));

	for(int j=0; j<ln;j++){
		*(unit+j)=*(vector+j)/norm;
	}

	return unit;
}

/* Function: isLinIndepenedent
 * Input: two vectors of equal length (not necessarily normalized)
 * Output: boolean that indicates if the two vectors are linearly independent
 */
bool Tableau::isLinIndependent(double* vect1, double* vect2){
	int ln = sizeof vect1/sizeof vect1[0];
	bool match=true;

	double* uv1= makeUnitVector(vect1);
	double* uv2= makeUnitVector(vect2);

	//check for match
	for(int i=0;i<ln;i++){
		match=match&&(*(uv1+i)!=*(uv2+i));
	}

	return match;
}

// b_vect is the last column of the matrix minus the first element
bool Tableau::makesNegativeSoln(double* sample_basis, double* b_vect){


	// define invBase:=get inverted form <- ACML
	// calculate matrix-vector product invBase*b_vect <- maybe ACML
	// return: does product have at least one negative element?

	return false;
}

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
 * NOTE2: This is definitely not the efficient way to implement this
 */
int* Tableau::getFeasibleIncoming(int* candidates, int* feasibles){

	int ln = sizeof current_basis/sizeof current_basis[0];
	int lnc = sizeof candidates/sizeof cadidates[0];

	// define & malloc int* inFeasible
	int* inF=malloc();

	// define & malloc int* outFeasible
	int* outF=malloc();

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
				double* sample= makeSampleBase(current_basis, candidate+j, i);
				if(!makesNegativeSoln(sample,getB(matrix))){
					inF+counter=candidates+j;
					outF+counter=current_basis+i;
				}
			}

		}
	}

	return null;
}

// assumes column numbers begin with 0
double* Tableau::getCol(int colNum){
	int ln = sizeof current_basis/sizeof current_basis[0];

	double* col=malloc(ln*sizeof(double));
	for(int i=0;i<ln;i++){
		col+i=matrix[i][colNum];
	}
	return col;
}

double* Tableau::getB(){
	return (getCol(width()-1))+1;
}
