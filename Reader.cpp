#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <stdlib.h>
#include <vector>
#include "GridPos.cpp"

using namespace std;


int main() {
	ifstream file;
	file.open("TESTPROB");

	string s;

	int numRows = 1;
	int numCols = 1; //numCols is offset by 1, col 1 means its actually col 0
	map<string, int> rowMap;
	map<string, int> colMap;
	vector<GridPos> matrixPositions;

	if (file.is_open()) {
		getline(file, s);
		while (!file.eof()){

			if (s.compare("ROWS") == 0){
				//do stuff for ROW
				getline(file, s);
				while(s[0] == ' '){
					if (s[1] != 'N'){
						rowMap[s.substr(4, 10)] = numRows;
						numRows++;
					} else {
						rowMap[s.substr(4, 10)] = 0;
					}
					getline(file, s);
				}
			} else if (s.compare("COLUMNS") == 0){
				//do stuff for COLUMNS
				getline(file, s);
				while (s[0] == ' '){
					//get the column (variable name)
					string colName = s.substr(4, 10);
					//strip off whitespace at the end
					size_t lastChar = colName.find_last_not_of(" ");
					if (lastChar != string::npos){
						colName.erase(lastChar+1);
					}
					
					//put the column into the map if it doesn't already exist
					if (colMap[colName] == 0){
						colMap[colName] = numCols;
						numCols++;
					}

					//get the row
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
						rowName = s.substr(39, 10);
						//strip off whitespace at the end
						lastChar = rowName.find_last_not_of(" ");
						if (lastChar != string::npos){
							rowName.erase(lastChar+1);
						}
						val = atof (s.substr(50, 15).c_str());

						GridPos pos2 (rowMap[rowName], colMap[colName] - 1, val);
						matrixPositions.push_back(pos2);
					}
					
					getline(file, s);

				}
			} else {
				getline(file, s);
			}
		}
	}
	file.close();
	
	for (int i = 0; i < matrixPositions.size(); i++){
		cout << matrixPositions[i].row << '\t' << matrixPositions[i].col << '\n';
	}
	return 0;
}