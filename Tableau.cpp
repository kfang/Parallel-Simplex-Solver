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
