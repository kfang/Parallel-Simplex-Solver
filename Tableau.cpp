#include <iostream>
using namespace std;

class Tableau {
	double** matrix;
	int rows;
	int cols;
	int* current_basis;
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

int* Tableau::get_candidate_cols(int *current_basis, int numVars){

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

bool Tableau::isLinIndependent(double* vect1, double* vect2){
	int ln = sizeof vect1/sizeof vect1[0];
	bool match=true;

	//check for match
	for(int i=0;i<ln;i++){
		match=match&&(*(vect1+i)!=*(vect2+i));
	}

	return match;
}

// b_vect is the last column of the matrix minus the first element
bool Tableau::makesNegativeSoln(double* sample_basis, double* b_vect){


	// define invBase:=get inverted form <- BLAS

	// calculate matrix-vector product invBase*b_vect <- maybe BLAS

	// return: does product have negative elements?

	return false;
}

bool Tableau::getFeasibleIncoming(int* candidates, int* current_basis){

	// (cont) means 'previous comment line continues here'
	//
	// define & malloc int* inFeasible
	// define & malloc int* outFeasible
	// for each outCol in current_basis
	// define double* sub_basis:= current_basis without outCol
	// 		for each inCol in candidates
	//			for each baseCol in sub_basis
	//				define bool indep: =accumulate (logical add)
	//				(cont)check isLinIndepenedent(baseCol, inCol)
	// 			if indep then
	// 				create sample_basis:=sub_basis with inCol
	//				define bool nonNegative: = check
	//  			(cont)makesNegativeSoln(sample_basis)
	// 			if nonNegative then
	//				add to inFeasible inCol
	//				add to outFeasible outCol
	// if inFeasible is not empty
	// 		return true

	return false;
}
