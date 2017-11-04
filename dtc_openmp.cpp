#include <omp.h>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <cstdlib>
#include <climits>

#define trainingData "hayes-roth.data.txt"
#define testingData "hayes-roth.data.txt"

using namespace std;

vector <vector <int> > fileContent;
vector <vector <int> > testFileContent;

int numOfAttrib, numOfDataEle;

struct Node{
	int numOfChildren;
	int val;
	int branchVal;
	int attribute;
	struct Node *child[10];
};

typedef struct Node node;

node* create(){
	node* n = new node;
	n->numOfChildren = 0;
	n->attribute = -1;
	n->val = -1;
	n->branchVal = -1;
	return n;
}

void readCSV(string str)
{
	if(str.compare("training")==0){
		ifstream ifs(trainingData);
		string line;

		while(getline(ifs,line)){
			stringstream lineStream(line);
			string cell;
			vector <int> values;
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

		while(getline(ifs,line)){
			stringstream lineStream(line);
			string cell;
			vector <int> values;
			while(getline(lineStream,cell,',')){
				const char *cstr = cell.c_str();
				values.push_back(atoi(cstr));
			}
			testFileContent.push_back(values);
		}
		ifs.close();
	}
}

double infoGain(int attr,vector <int> data)
{
	int i,branchVal,dataSize,subDataValue;
	
}

int select(vector <int> &attr,vector <int> data)
{
	int i,splitAttr;
	vector <double> iGain(attr.size(),0);
	double maxIGain;
	maxIGain = INT_MIN;
	for(i=1;i<attr.size()-1;i++){
		if(attr[i]==0){
			#pragma omp task
			iGain[i] = infoGain(i,data)
		}
	}

	#pragma omp taskwait
	for(i=1;i<attr.size()-1;i++){
		if(iGain > maxIGain){
			maxIGain = iGain;
			splitAttr = i;
		}
	}

	if(maxIGain == INT_MIN){
		return -1;
	}
	attr[splitAttr]=1;
	return splitAttr;
}

int popularVote(vecotr<int> data)
{
	int i,outputClass,ans,maxVal;
	map <int, int> dataCount;
	map <int, int>::iterator it;

	#pragma omp parallel for
	for(i=0;i<data.size();i++){
		outputClass = fileContent[data[i]][numOfAttrib-1];
		#pragma omp critical
		{
			if(dataCount.find(outputClass) == dataCount.end()){
				dataCount.insert(make_pair(outputClass,1));
			}
			else{
				dataCount[outputClass]++;
			}
		}
	}
	maxVal = INT_MIN;

	for(it=dataCount.begin();it!=dataCount.end();it++){
		if(it->second > maxVal){
			ans = it->first;
		}
	}
	return ans;
}

void decision(vector<int> attr,vector<int> data,node *root)
{
	int flag,selectedAttribute,i;
	if(data.size()==0){
		return;
	}
	flag=1;
	
	#pragma omp parallel for shared(flag) private(i)
	for(i=1;i<data.size();i++){
		if(fileContent[data[i]][numOfAttrib-1]!=fileContent[data[i-1]][numOfAttrib-1]){
			flag=0;
		}
	}
	
	if(flag==1){
		root->val=fileContent[data[0]][numOfAttrib-1];
		return;
	}
	
	selectedAttribute = select(attr,data);
	root->attribute = selectedAttribute;
	
	if(selectedAttribute == -1){
		root->val = popularVote(data);
		return;
	}

	map<int, vector <int> > dividedData;
	map<int, vector <int> >::iterator it;
	int attrVal;

	#pragma omp parallel for
	for(i=0;i<data.size();i++){
		attrVal = fileContent[data[i]][selectedAttribute];
		#pragma omp critical
		{
			if(dividedData.find(attrVal) == dividedData.end()){
				vector <int> x;
				x.push_back(data[i]);
				dividedData.insert(make_pair(attrVal,x));
			}
			else{
				dividedData[attrVal].push_back(data[i]);
			}
		}
	}

	for(i=0,it=dividedData.begin();it!=dividedData.end();it++,i++){
		root->numOfChildren++;
		node* childNode;
		childNode = create();
		childNode->branchVal = it->first;
		root->child[i] = childNode;

		#pragma omp task
		decision(attr, it->second, childNode);
	}

}

int main()
{
	omp_set_dynamic(1);
	omp_set_nested(1);

	int i;
	node * root;

	vector <int> data;
	vector <int> attr;

	readCSV("training");

	numOfAttrib = fileContent[0].size();
	numOfDataEle = fileContent.size();

	#pragma omp parallel shared(numOfDataEle,numOfAttrib,data,attr) private(i) num_threads(2)
	{
		#pragma omp single
		{
			for(i=0;i<numOfDataEle;i++){
				data.push_back(i);
			}
		}
		#pragma omp single
		{
			for(i=0;i<numOfAttrib;i++){
				attr.push_back(0);
			}
		}
	}

	root = create();
	#pragma omp parallel
	{
		#pragma omp single
		{
			#pragma omp task 
			decision(attr,data,root);
		}
	}

	return 0;
}