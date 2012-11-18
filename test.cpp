#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <stdio.h>
#include "TableauTest.cpp"
/*
 * This calls TableauTest.
 */
using namespace std;

int main(){

	printf("\n starting..\n\n");
	TableauTest tt=new TableauTest(true);
	tt.printMat();
	printf("\n width =%d, and height=%d\n\n", tt.width(), tt.height());
	int* cands=tt.get_candidate_cols();
	int candSize=tt.width()-tt.height();

	for(int i=0; i<candSize; i++){
		printf("\n candidate #%d = %d\n\n",i,cands[i]);
	}

	tt.update_candidate_cols(cands);
	cands=tt.get_candidate_cols();

	for(int i=0; i<candSize; i++){
			printf("\n candidate #%d = %d\n\n",i,cands[i]);
	}

	double* testVect=new double[7];

	for(int i=0;i< 7;i++){
		testVect[i]=i+1;
	}

	int ln = sizeof(*testVect)/sizeof(testVect[0]);
	printf("\n testsize=%f", testVect[2]);

	double* result=tt.makeUnitVector(testVect);

	for(int i=0; i<7; i++){
			printf("\n new vector elem#%d = %f\n\n",i,result[0]+i);
	}
}


