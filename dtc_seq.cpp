#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

#define fileName "spect.csv"

using namespace std;

vector <vector <string> > readCSV()
{
	vector <vector <string> > fileContent;

	ifstream ifs(fileName);
	string line;

	while(getline(ifs,line)){
		stringstream lineStream(line);
		string cell;
		vector <string> values;
		while(getline(lineStream,cell,',')){
			values.push_back(cell);
		}
		fileContent.push_back(values);
	}

	ifs.close();

	return fileContent;

}

int main()
{
	vector <vector <string> > fileContent;
	fileContent = readCSV();
	// int x,y,i,j;
	// x=fileContent.size();
	// y=fileContent[0].size();
	// for(i=0;i<x;i++){
	// 	for(j=0;j<y;j++){
	// 		cout << fileContent[i][j] << " ";
	// 	}
	// 	cout << endl;
	// }
	
	return 0;
}