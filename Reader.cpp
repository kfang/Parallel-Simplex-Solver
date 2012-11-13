#include <iostream>
#include <fstream>
#include <string>
#include <map>

using namespace std;

int main() {
	ifstream file;
	file.open("TESTPROB");

	string s;

	int numRows = 1;
	int numCols = 0;
	map<string, int> rowMap;
	map<string, int> colMap;

	if (file.is_open()) {
		while (!file.eof()){

			getline(file, s);
			cout << s.compare("ROWS");
			cout << '\n';
			if (s.compare("ROWS") == 0){
				//do stuff for ROW
				getline(file, s);
				while(s[0] == ' '){
					cout << s;
					cout << '\n';
					if (s[1] != 'N'){
						cout << s.substr(4, 10);
						rowMap.insert(pair<string, int>(s.substr(4, 10), numRows));
						numRows++;
					} else {
						rowMap.insert(pair<string, int>(s.substr(4, 10), 0));
					}
					getline(file, s);
				}
			}
		}
	}
	file.close();
	cout << numRows;
	return 0;
}