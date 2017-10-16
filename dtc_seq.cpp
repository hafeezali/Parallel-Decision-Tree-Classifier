#include <vector>
#include <map>
#include <queue>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <set>
#include <climits>

// filename of training data and testing data
#define trainingData "hayes-roth.data.txt"
#define testingFile "hayes-roth.test.txt"

using namespace std;

// 2d vector to store training and testing data
vector <vector <int> > fileContent;

int numOfAttrib, numOfDataEle;
double infoGainOfData;

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

// function to read data and store in fileContent vector(2d)
void readCSV()
{
	// input file stream (ifs) for reading data from file
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

// function to calculate entropy 
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
	// Entropy E = (a/(a+b+...))*(log(a/(a+b+...))/log(2)) + (b/(a+b+...))*(log(b/(a+b+...))/log(2)) + ...
	for(i=0;i<counts.size();i++){
		entropy += ((counts[i]/total)*(log(counts[i]/total)/log(2)));
	}
	return entropy;
}

// function to calculate information gain
// attr: attribute for which gin must be calculated
// data: data row nos(in the file and index in "fileContent" vector) used for calculating information gains
double infoGain(int attr,vector <int> data)
{
	int i,branchVal,dataSize,subDataValue;
	double attrInfoGain;
	// branchCount: count of each attribute value
	map<int, int> branchCount;
	map<int, int>::iterator branchCountIT;
	// dataElements[i]: vector containing all data elements having attribute value "i"
	map<int, vector<int> > dataElements;
	for(i=0;i<data.size();i++){
		branchVal = fileContent[i][attr];
		if(branchCount.find(branchVal) == branchCount.end()){
			// if branchCount does not contain the key branchVal, then insert the pair(branchVal,1)
			branchCount.insert(make_pair(branchVal,1));
			vector <int> x;
			x.push_back(i);
			// add "i" to the vector containing all data elements whose attribute value is branchVal
			dataElements.insert(make_pair(branchVal,x));
		}
		else{
			branchCount[branchVal]++;
			dataElements[branchVal].push_back(i);
		}
	}
	attrInfoGain=0;
	dataSize=data.size();
	for(branchCountIT = branchCount.begin();branchCountIT!=branchCount.end();branchCountIT++){
		vector <int> subData = dataElements[branchCountIT->first];
		// subDataCounts: contains count of data elements belonging to the different output classes
		map <int, int> subDataCounts;
		map <int, int>::iterator subDataCountsIT;
		for(i=0;i<subData.size();i++){
			subDataValue = fileContent[subData[i]][numOfAttrib-1];
			if(subDataCounts.find(subDataValue) == subDataCounts.end()){
				// if subDataCounts does not contain subDataValue as key, then insert the pair(subDataValue,1)
				subDataCounts.insert(make_pair(subDataValue,1));
			}
			else{
				// if it contains the key, then increment count
				subDataCounts[subDataValue]++;
			}
		}
		// subDataCountsArr: contains all counts of each output class value
		vector <double> subDataCountsArr;
		for(subDataCountsIT=subDataCounts.begin();subDataCountsIT!=subDataCounts.end();subDataCountsIT++){
			subDataCountsArr.push_back((double)subDataCountsIT->second);
		}
		attrInfoGain+= (branchCountIT->second/dataSize)*entropy(subDataCountsArr);
	}
	return infoGainOfData - attrInfoGain;
}

// function to get information gain of training data
void getInfoGainOfData()
{
	int i,classVal;
	// classCount: keeps a count of the number of data points belonging to a particular class
	map<int, int> classCount;
	map<int, int>::iterator it;
	// counts: store all the counts of all the classes, used for calculating entropy
	vector<double> counts;
	for(i=0;i<fileContent.size();i++){
		classVal = fileContent[i][numOfAttrib-1];
		// result->second = false if insert failed
		// insert operation fails if "classVal" key already present in map "classCount"
		if(classCount.find(classVal) == classCount.end()){
			classCount.insert(make_pair(classVal,1));
		}
		else{
			classCount[classVal]++;
		}
	}
	for(it=classCount.begin();it!=classCount.end();it++){
		counts.push_back((double)it->second);
	}
	infoGainOfData = entropy(counts);
}

// function to determine the splitting attribute
// attr: candidate attributes for splitting attribute, attr[i]=1 if already used
// data: data row nos(in the file and index in "fileContent" vector) used for calculating information gains
int select(vector <int> &attr,vector <int> data)
{
	int i,splitAttr;
	double iGain,maxIGain;
	maxIGain = INT_MIN;
	for(i=1;i<attr.size()-1;i++){
		if(attr[i]==0){
			iGain = infoGain(i,data);
			if(iGain>maxIGain){
				// store maximum information gain value along with attribute 
				maxIGain = iGain;
				splitAttr = i;
			}
		}
	}
	if(maxIGain==INT_MIN){
		return -1;
	}
	// mark splitAttr as used
	attr[splitAttr]=1;
	return splitAttr;
}

// function for returning most probable output class
int popularVote(vector<int> data)
{
	
}

// builder function for generating decision tree
// attr: candidate attributes for splitting attribute, attr[i]=1 if already used
// data: data row nos(in the file and index in "fileContent" vector) used for calculating information gains
void decision(vector<int> attr,vector<int> data,node *root)
{
	int flag,selectedAttribute,numOfAttribValues,i;
	// if no data then can't decide the class value, must take popular vote
	if(data.size()==0){
		return;
	}
	flag=1;
	for(i=1;i<data.size();i++){
		if(fileContent[data[i]][numOfAttrib-1]!=fileContent[data[i-1]][numOfAttrib-1]){
			// flag = 0 if data contains more than one class value
			flag=0;
			break;
		}
	}
	// flag = 1 if all the data belong to the same class
	if(flag==1){
		// assign class value to node and return
		root->val=fileContent[data[0]][numOfAttrib-1];
		return;
	}
	// selectedAttribute : splitting attribute
	selectedAttribute=select(attr,data);
	root->attribute = selectedAttribute;

	if(selectedAttribute == -1){
		// running out of attributes
		root->attribute = popularVote(data);
		return;
	}

	// dividedData: divide data and store based on attribute values
	map<int, vector <int> > dividedData;
	map<int, vector <int> >::iterator it;
	int attrVal;

	for(i=0;i<data.size();i++){
		attrVal = fileContent[data[i]][selectedAttribute];
		if(dividedData.find(attrVal) == dividedData.end()){
			// if attrVal not present as key in dividedData, then insert pair (attrVal,x), where x is a vector
			vector <int> x;
			x.push_back(i);
			dividedData.insert(make_pair(attrVal,x));
		}
		else{
			// if attrVal is present, add "i" to the corresponding vector
			dividedData[attrVal].push_back(i);
		}
	}
	for(i=0,it=dividedData.begin();it!=dividedData.end();it++,i++){
		// create childNode and recurse on it
		root->numOfChildren++;
		node* childNode;
		childNode = create();
		childNode->branchVal = it->first;
		root->child[i] = childNode;
		decision(attr, it->second, childNode);
	}

}

// function for printing and debugging decision tree : bfs traversal
void printDecisionTree(node *root)
{
	queue <node> bfsQ;
	// i : indicates tree level
	int i,x,j;
	node* nextNode;
	bfsQ.push(*root);
	i=0;
	cout << "Level " << i << ":" << endl;
	cout << root->attribute << " "<< endl;
	while(bfsQ.size()!=0){
		nextNode = &(bfsQ.front());
		bfsQ.pop();
		x = nextNode->numOfChildren;
		i++;
		j=0;
		cout << "Level " << i << ":" << endl;
		while(j<x){
			bfsQ.push(*(nextNode->child[j]));
			cout << nextNode->child[j]->attribute << " ";
			j++;
		}
		cout << endl;
	}
	return;
}

int main()
{
	int i;
	node* root;
	// vector to store row number for data in file
	vector <int> data;
	// vector to check if attribute has already been used or not
	vector <int> attr;

	readCSV();
	numOfAttrib = fileContent[0].size()-2;
	numOfDataEle = fileContent.size();
	getInfoGainOfData();

	for(i=0;i<numOfDataEle;i++){
		data.push_back(i);
	}
	for(i=0;i<=numOfAttrib+2;i++){
		attr.push_back(0);
	}

	// create decision tree
	root = create();
	decision(attr,data,root);
	printDecisionTree(root);

	return 0;
}