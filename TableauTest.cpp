#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <stdio.h>


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
	TableauTest (bool);
	int height(){return rows;};
	int width(){return cols;};
	void printMat();

	void update_candidate_cols(int* oldcands);
	int* get_candidate_cols();
	void makeUnitVector(double* vector, double* unit);
	bool isLinIndependent(double* vect1, double* vect2);

};


TableauTest::TableauTest(bool example){
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

int* TableauTest::get_candidate_cols(){
	return candidates;
}

void TableauTest::update_candidate_cols(int* oldcands){

	int numVars=cols-1;

	// how many variables aren't already within the current basis?

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

void TableauTest::makeUnitVector(double* vector, double* unit){

	double norm=0;

	// get norm
	for(int i=0; i<cols-1;i++){
		double x=vector[i];
		double y=pow(x,2);
		norm=norm+y;
	}

	norm=sqrt(norm);

	//double unit[cols-1];

	for(int j=0; j<cols-1;j++){
		unit[j]=(vector[j]/norm);
		//printf("\nunit[%d]=%f\n",j,unit[j]);
	}


	//return &unit[0];
}
// Tested
/* Function: isLinIndepenedent
 * Input: two vectors of equal length (not necessarily normalized)
 * Output: boolean that indicates if the two vectors are linearly independent
 */
bool TableauTest::isLinIndependent(double* vect1, double* vect2){
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
		printf("\n matching %f with %f\n",uv1[i], uv2[i]);
		noMatch=noMatch||(uv1[i]!=uv2[i]);

	}

	free(uv1);
	free(uv2);
	return noMatch;
}
