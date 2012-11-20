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

	/*
	 * Test get_candidate_cols
	 */
	int* cands=tt.get_candidate_cols();
	int candSize=tt.width()-tt.height();

	for(int i=0; i<candSize; i++){
		printf("\n candidate #%d = %d\n\n",i,cands[i]);
	}

	/*
	 * Test update_candidate_cols
	 */
	tt.update_candidate_cols(cands);
	cands=tt.get_candidate_cols();

	for(int i=0; i<candSize; i++){
			printf("\n candidate #%d = %d\n\n",i,cands[i]);
	}

	/*
	* Test makeUnitVector
	*/

	double* testVect=new double[7];
	double* unitVect=new double[7];

	for(int i=0;i< 7;i++){
		testVect[i]=i+1;
		unitVect[i]=0;
	}

	tt.makeUnitVector(testVect,unitVect);

	for(int i=0; i<7; i++){
			printf("\n new vector elem#%d = %f\n\n",i,unitVect[i]);
	}

	//test isLinIndependent
	double* testVect2=new double[7];

	for(int i=0;i< 7;i++){
		testVect2[i]=2*(i+1);
	}

	bool test1=tt.isLinIndependent(testVect, testVect2);
	printf("\n test1 s/b false and is = %s\n\n",(test1)?"true":"false");

	double* testVect3=new double[7];
	for(int i=0;i< 7;i++){
		if(i!=3){
			testVect3[i]=2*(i+1);
		}else{
			testVect3[i]=11*(i);
			printf("\n this val = %f\n", testVect3[i]);
		}
	}

	bool test2=tt.isLinIndependent(testVect, testVect3);
	printf("\n test2 s/b true and is = %s\n\n",(test2)?"true":"false");

	//test getSlice
	double* emptyVect=new double[7];

	for(int i=0;i< 7;i++){
			emptyVect[i]=0;
	}

	double* testCol=tt.getSlice(2,emptyVect, true);

	for(int i =0; i<tt.width()-1;i++){
		printf("\n elem# %d= %f\n", i, testCol[i]);
	}

	//test getB

	emptyVect=new double[3];

	double* testB=tt.getB(emptyVect);

	printf("\n");
	for(int i =0; i<tt.height()-1;i++){
			printf("\n elem# %d= %f\n", i, testB[i]);
		}
}


