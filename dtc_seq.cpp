#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <set>

#define fileName "hayes-roth.data.txt"

using namespace std;

struct node{
	int val;
	int attribute;
	node *child[10];
}

node* create(){
	node* n = new node;
	n->attribute = -1;
	n->val = -1;
	return n;
}

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

vector <int> getCardinality(vector< vector <string> > fileContent)
{
	vector <int> cardinality(fileContent[0].size(),0);
	int i,j;
	for(i=1;i<fileContent[0].size()-1;i++){
		set <int> values;
		for(j=0;j<fileContent.size();j++){
			values.insert(fileContent[j][i]);
		}
		cardinality[i]=set.size();
	}
	return cardinality;
}

int main()
{
	vector <vector <string> > fileContent;
	vector <int> cardinality;
	int numOfAttrib, numOfDataEle;
	node* root;
	
	fileContent = readCSV();
	numOfAttrib = fileContent[0].size()-2;
	numOfDataEle = fileContent.size();
	cardinality = getCardinality(fileContent);

	root = create();
	decision(,root);
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