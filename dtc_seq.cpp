#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <set>

#define fileName "hayes-roth.data.txt"

using namespace std;

vector <vector <string> > fileContent;
vector <int> cardinality;
int numOfAttrib, numOfDataEle;

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

void readCSV()
{
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
}

void getCardinality()
{
	vector <int> cardinal(fileContent[0].size(),0);
	int i,j;
	for(i=1;i<fileContent[0].size()-1;i++){
		set <int> values;
		for(j=0;j<fileContent.size();j++){
			values.insert(fileContent[j][i]);
		}
		cardinal[i]=set.size();
	}
	cardinality=cardinal;
}

int select(vector <int> attr,vector <int> data)
{
	int i;
	for(i=0;i<attr.size();i++){
		
	}
}

void decision(vector<int> attr,vector<int> data,node *root)
{
	int flag,selectedAttribute;

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
		root->val=fileContent[data[0][numOfAttrib-1]];
		return;
	}
	selectedAttribute=select(attr,data);
}

int main()
{
	int i;
	node* root;
	vector <int> data;
	vector <int> attr;

	readCSV();
	numOfAttrib = fileContent[0].size()-2;
	numOfDataEle = fileContent.size();
	getCardinality();

	for(i=0;i<=numOfDataEle;i++){
		data.push_back(i);
	}
	for(i=0;i<=numOfAttrib;i++){
		attr.push_back(0);
	}

	root = create();
	decision(attr,data,root);

	return 0;
}