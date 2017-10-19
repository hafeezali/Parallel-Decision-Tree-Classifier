#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "math.h"
#include "limits.h"

#define trainingData "hayes-roth.data.txt"
#define testingData "hayes-roth.data.txt"

using namespace std;

vector <vector <int> > trainFile;
vector <vector <int> > testFile;

int numOfAttrib, numofDataEle;

// node structure of the decision tree
// attribute: splitting attribute (= -1 if leaf node)
// val: class value at leaf node (= -1 if decision node)
// branchVal: make branch decision based on this value
struct Node{
	int numOfChildren;
	int val;
	int branchVal;
	int attribute;
	struct Node *child[10];
};

typedef struct Node node;

// initialising tree node
node* create(){
	node* n = new node;
	n->numOfChildren = 0;
	n->attribute = -1;
	n->val = -1;
	n->branchVal = -1;
	return n;
}

// function to read data and store in fileContent & testFileContent vectors(2d)
void readCSV(string str)
{
	// input file stream (ifs) for reading data from file
	if(str.compare("training")==0){
		ifstream ifs(trainingData);
		string line;

		// read from ifs into string 'line'
		while(getline(ifs,line)){
			stringstream lineStream(line);
			string cell;
			vector <int> values;
			// collecting row data from file delimited by ','
			while(getline(lineStream,cell,',')){
				const char *cstr = cell.c_str();
				values.push_back(atoi(cstr));
			}
			fileContent.push_back(values);
		}
		ifs.close();
	}
	else if(str.compare("testing")==0){
		ifstream ifs(testingData);
		string line;
		
		// read from ifs into string 'line'
		while(getline(ifs,line)){
			stringstream lineStream(line);
			string cell;
			vector <int> values;
			// collecting row data from file delimited by ','
			while(getline(lineStream,cell,',')){
				const char *cstr = cell.c_str();
				values.push_back(atoi(cstr));
			}
			testFileContent.push_back(values);
		}
		ifs.close();
	}
}

void decision(vector<int> attr, vector<int> data, node* root)
{
	int flag,selectedAttribute,i;
	if(data.size()==0){
		return;
	}
	flag=1;
	for(i=1;i<data.size();i++){
		if(fileContent[data[i]][numOfAttrib-1]!=fileContent[data[i-1]][numOfAttrib-1]){
			flag=0;
			break;
		}
	}
	if(flag==1){
		root->val=fileContent[data[0]][numOfAttrib-1];
		return;
	}
	
}

int main()
{
	int i;
	node* root;

	readCSV("training");

	numOfAttrib = fileContent[0].size();
	numOfDataEle = fileContent.size();

	int attr[numOfAttrib], data[numOfAttrib];

	for(i=0;i<numOfDataEle;i++){
		data[i]]=i;
	}
	for(i=0;i<numOfAttrib;i++){
		attr[i]=0;
	}

	root = create();
	decision(attr,data,root);

	return 0;
}