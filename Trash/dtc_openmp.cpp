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

double entropy(vector <double> counts)
{
	double total,entropy;
	int i;
	total=0;
	for(i=0;i<counts.size();i++){
		if(counts[i]==0){
			return 0;
		}
		total+=counts[i];
	}
	entropy=0;
	for(i=0;i<counts.size();i++){
		entropy += (counts[i]/total)*(log(counts[i]/total)/log(2));
	}
	return -1 * entropy;
}

double getInfoGainOfData(vector <int> data)
{
	int i,classVal;
	map<int, int> classCount;
	map<int, int>::iterator it;

	vector<double> counts;
	#pragma omp parallel for
	for(i=0;i<data.size();i++){
		classVal = fileContent[data[i]][numOfAttrib-1];
		#pragma omp critical
		{
			if(classCount.find(classVal) == classCount.end()){
				classCount.insert(make_pair(classVal,1));
			}
			else{
				classCount[classVal]++;
			}
		}
	}
	for(it=classCount.begin();it!=classCount.end();it++){
		counts.push_back((double)it->second);
	}
	return entropy(counts);
}

double infoGain(int attr,vector <int> data)
{
	int i,branchVal,dataSize,subDataValue;
	double attrInfoGain;

	map<int, int> branchCount;
	map<int, int>::iterator branchCountIT;

	map<int, vector<int> > dataElements;
	#pragma omp parallel for
	for(i=0;i<data.size();i++){
		branchVal = fileContent[data[i]][attr];
		#pragma omp critical
		{
			if(branchCount.find(branchVal) == branchCount.end()){
				branchCount.insert(make_pair(branchVal,1));
				vector <int> x;
				x.push_back(data[i]);
				dataElements.insert(make_pair(branchVal,x));
			}
			else{
				branchCount[branchVal]++;
				dataElements[branchVal].push_back(data[i]);
			}
		}
	}
	attrInfoGain=0;
	dataSize=data.size();
	for(branchCountIT = branchCount.begin();branchCountIT!=branchCount.end();branchCountIT++){
		vector <int> subData = dataElements[branchCountIT->first];

		map <int, int> subDataCounts;
		map <int, int>::iterator subDataCountsIT;
		#pragma omp parallel for
		for(i=0;i<subData.size();i++){
			subDataValue = fileContent[subData[i]][numOfAttrib-1];
			#pragma omp critical
			{
				if(subDataCounts.find(subDataValue) == subDataCounts.end()){
					subDataCounts.insert(make_pair(subDataValue,1));
				}
				else{
					subDataCounts[subDataValue]++;
				}
			}
		}
		vector <double> subDataCountsArr;
		for(subDataCountsIT=subDataCounts.begin();subDataCountsIT!=subDataCounts.end();subDataCountsIT++){
			subDataCountsArr.push_back((double)subDataCountsIT->second);
		}
		attrInfoGain+= ((double)branchCountIT->second/(double)dataSize)*entropy(subDataCountsArr);
	}
	return getInfoGainOfData(data) - attrInfoGain;
}


//problem with this
int select(vector <int> &attr,vector <int> data)
{
	int i,splitAttr;
	vector <double> iGain(attr.size(),0);
	double maxIGain;
	maxIGain = INT_MIN;
	for(i=1;i<attr.size()-1;i++){
		if(attr[i]==0){
			#pragma omp task
			iGain[i] = infoGain(i,data);
		}
	}

	#pragma omp taskwait
	for(i=1;i<attr.size()-1;i++){
		if(iGain[i] > maxIGain){
			maxIGain = iGain[i];
			splitAttr = i;
		}
	}

	if(maxIGain == INT_MIN){
		return -1;
	}
	attr[splitAttr]=1;
	return splitAttr;
}

int popularVote(vector<int> data)
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
	#pragma omp taskwait

}

// function for printing and debugging decision tree : bfs traversal
void printDecisionTree(node *root)
{
	printf("Printing decision tree:\n");
	queue <node> bfsQ;
	int x,j;
	node* nextNode;
	bfsQ.push(*root);
	cout << root->attribute << endl;
	// implementing bfs traversal of tree
	while(bfsQ.size()!=0){
		nextNode = &(bfsQ.front());
		bfsQ.pop();
		x = nextNode->numOfChildren;
		j=0;
		while(j<x){
			bfsQ.push(*(nextNode->child[j]));
			cout << nextNode->child[j]->attribute << " ";
			j++;
		}
		cout << endl;
	}
	return;
}

// function for testing decision tree
void test(node* root)
{
	int i,pos,neg,noResult,attr,attrVal,j,flag;
	node* temp;
	pos=0;
	neg=0;
	noResult=0;
	readCSV("testing");
	for(i=0;i<testFileContent.size();i++){
		temp=root;
		flag=0;
		//traverse decision tree
		while(temp->val==-1 && temp->attribute!=-1){
			attr = temp->attribute;
			attrVal=testFileContent[i][attr];
			for(j=0;j<temp->numOfChildren;j++){
				if(temp->child[j]->branchVal == attrVal){
					break;
				}
			}
			if(j==temp->numOfChildren){
				flag=1;
				break;
			}
			else{
				temp=temp->child[j];
			}
		}
		if(temp->val == testFileContent[i][numOfAttrib-1]){
			// predicted value = actual value
			pos++;
		}
		else{
			// predicted value != actual value
			neg++;
		}
		if(temp->val == -1 || flag==1){
			// no predicted value
			noResult++;
		}
	}
	cout << "Positive: " << pos << endl;
	cout << "Negative: " << neg << endl;
	cout << "No Result: " << noResult << endl;

	return;
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
	#pragma omp parallel num_threads(8)
	{
		#pragma omp single
		{
			decision(attr,data,root);
		}
	}

	printDecisionTree(root);

	test(root);

	return 0;
}