#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <stdlib.h>
#include <vector>
#include "GridPos.cpp"
#include "Tableau.cpp"

using namespace std;


void doSlack(Tableau* tableau){
	//store old number of columns, will have to use
	//this number to move the RHS column to the end
	int oldColNum = tableau->width() - 1;

	//add columns equals to the number of rows - 1 
	tableau->addCol(tableau->height() - 1);

	//move the RHS column to the end
	tableau->swapCols(oldColNum, tableau->width() - 1);

	//place in the swaps, starts at row=1, col=oldColNum
	for (int row = 1; row < tableau->height(); row++){
		(*tableau)[row][oldColNum + row - 1] = 1.0;
	}

	//you now have a standardized tableau matrix!
}

int main() {
	ifstream file;
	file.open("TESTPROB");

	string s;

	int numRows = 1; //starts with 1 since row 0 is the objective function
	int numCols = 1; //numCols is offset by 1, col 1 means its actually col 0
	map<string, int> rowMap; //stores a map from the row name to its row number
	map<string, int> colMap; //stores a map from the col name to its col number
	vector<GridPos> matrixPositions; //a list of positions and values for the tableau

	if (file.is_open()) {
		getline(file, s);
		while (!file.eof()){

			if (s.compare("ROWS") == 0){
				//do stuff for ROW
				getline(file, s);
				while(s[0] == ' '){
					//look for the objective function
					if (s[1] != 'N'){
						//just a regular row
						rowMap[s.substr(4, 10)] = numRows;
						numRows++;
					} else {
						//the objective function, place it in row 0
						rowMap[s.substr(4, 10)] = 0;
					}
					getline(file, s);
				}
			} else if (s.compare("COLUMNS") == 0){
				//do stuff for COLUMNS
				getline(file, s);
				
				while (s[0] == ' '){{
					//get the column (variable name)
					string colName = s.substr(4, 10);

					//strip off whitespace at the end
					size_t lastChar = colName.find_last_not_of(" ");
					if (lastChar != string::npos){
						colName.erase(lastChar+1);
					}
					
					//put the column into the map if it doesn't already exist
					//**you know it doesn't exist if the value of the key is 0
					//****yes, this is why the numCol is offset by 1 -_-;;
					if (colMap[colName] == 0){
						colMap[colName] = numCols;
						numCols++;
					}

					//get the row name
					string rowName = s.substr(14, 10);

					//strip off the whitespace at the end
					lastChar = rowName.find_last_not_of(" ");
					if (lastChar != string::npos){
						rowName.erase(lastChar+1);
					}

					//get the row/column value
					double val = atof (s.substr(24, 15).c_str());

					//We have all info to place data
					//generate a tuple containing (row, column, value)
					GridPos pos (rowMap[rowName], colMap[colName] - 1, val);
					matrixPositions.push_back(pos);

					//check for another set
					if (s.length() > 40){
						//get the row name
						rowName = s.substr(39, 10);

						//strip off whitespace at the end
						lastChar = rowName.find_last_not_of(" ");
						if (lastChar != string::npos){
							rowName.erase(lastChar+1);
						}
						
						//get the value
						val = atof (s.substr(50, 15).c_str());

						//generate the tuple, add it to the list
						GridPos pos2 (rowMap[rowName], colMap[colName] - 1, val);
						matrixPositions.push_back(pos2);
					}
					
					getline(file, s);}

				}
			} else if (s.compare("RHS") == 0){
				//do stuff for RHS
				getline(file, s);
				while (s[0] == ' '){
					//get the row name
					string rowName = s.substr(14, 10);

					//strip off whitespace
					size_t lastChar = rowName.find_last_not_of(" ");
					if (lastChar != string::npos){
						rowName.erase(lastChar+1);
					}

					//store the value
					double val = atof (s.substr(24, 15).c_str());

					//we know what row it goes in, have total number of columns
					//****remember that numCols is offset by +1
					//generate the tuple and add it to the list of (position, values)
					GridPos pos (rowMap[rowName], numCols - 1, val);
					matrixPositions.push_back(pos);

					//check for another value
					if (s.length() > 40){
						//get the row name
						rowName = s.substr(39, 10);

						//strip whitespace
						lastChar = rowName.find_last_not_of(" ");
						if (lastChar != string::npos){
							rowName.erase(lastChar+1);
						}

						//get the value
						val = atof (s.substr(50, 15).c_str());
						
						//generate the tuple and add it to the list
						GridPos pos2 (rowMap[rowName], numCols -1, val);
						matrixPositions.push_back(pos2);
					}
					getline(file, s);
				}
			} else {
				getline(file, s);
			}
		}
	} else {
		//if no file gets open, return -> we don't want to simplex on 
		//empty stuff... bad will happen
		return 1;
	}
	file.close();
	
	//-------------------------------------------
	//   CHANGING FREE VARIABLES NOT IMPLEMENTED
	//-------------------------------------------


	//Create the tableau object and populate it from the list of values created while
	//parsing the MPS file
	Tableau tableau (numRows, numCols);

	//inteate down the list (actually a vector) to populate the tableau with values
	for (int i = 0; i < matrixPositions.size(); i++){
		//this line looks scary, but its simply
		//tableau[row][col] = val;
		//where row and col come from the matrixPositions list
		tableau[matrixPositions[i].row][matrixPositions[i].col] = matrixPositions[i].val;
	}

	//fill in slack variables, we pass in the address since we want to actually modify
	//stuff in it at the address.
	doSlack (&tableau);

	//-------------------------------------------
	// ONCE ALL DATA IS IN TABLEAU, SHOULD CLEAN UP VARS
	// FROM READING IN FILE
	//-------------------------------------------
	//print for sanity check
	tableau.printMat();

	//
	return 0;
}
