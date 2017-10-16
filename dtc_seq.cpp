#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <set>

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
struct node{
	int val;
	int attribute;
	node *child[10];
}

// initialising tree node
node* create(){
	node* n = new node;
	n->attribute = -1;
	n->val = -1;
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
		vector <string> values;
		// collecting row data from file delimited by ','
		while(getline(lineStream,cell,',')){
			values.push_back(atoi(cell));
		}
		fileContent.push_back(values);
	}

	ifs.close();
}

		// void getCardinality()
		// {
		// 	vector <int> cardinal(fileContent[0].size(),0);
		// 	int i,j;
		// 	for(i=1;i<fileContent[0].size();i++){
		// 		set <string> values;
		// 		for(j=0;j<fileContent.size();j++){
		// 			values.insert(fileContent[j][i]);
		// 		}
		// 		cardinal[i]=set.size();
		// 	}
		// 	cardinality=cardinal;
		// }

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
	int i,branchVal,dataSize;
	double attrInfoGain;
	// branchCount: count of each attribute value
	map<int, int> branchCount;
	map<int, int>::iterator branchCountIT;
	// dataElements[i]: vector containing all data elements having attribute value "i"
	map<int, vector<int>> dataElements;
	for(i=0;i<data.size();i++){
		branchVal = fileContent[i][attr];
		pair<pair<int, int>::iterator, bool> result = branchCount.insert(make_pair(branchVal,1));
		if(result->second == false){
			result->first->second++;
		}
		vector <int> x;
		x.push_back(i);
		pair<pair<int, int>::iterator, bool> result = dataElements.insert(make_pair(branchVal,x));
		if(result->second == false){
			result->first->second.push_back(i);
		}
	}
	attrInfoGain=0;
	dataSize=data.size();
	for(branchCountIT = branchCount.begin();branchCountIT!=branchCount.end();branchCountIT++){
		vector <int> subData = dataElements[branchCountIT->first];
		map <int, int> subDataCounts;
		map <int, int>::iterator subDataCountsIT;
		for(i=0;i<subData.size();i++){
			subDataValue = fileContent[subData[i]][numOfAttrib-1];
			pair<pair<int, int>::iterator, bool> result = subDataCounts.insert(make_pair(subDataValue,1));
			if(result->second == false){
				result->first->second++;
			}
		}
		vector <int> subDataCountsArr;
		for(subDataCountsIT=subDataCounts.begin();subDataCountsIT!=subDataCounts.end();subDataCountsIT++){
			subDataCountsArr.push_back(subDataCountsIT->second);
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
	vector<int> counts;
	for(i=0;i<fileContent.size();i++){
		classVal = fileContent[i][numOfAttrib-1];
		// result->second = false if insert failed
		// insert operation fails if "classVal" key already present in map "classCount"
		pair<pair<int, int>::iterator, bool> result =  classCount.insert(make_pair(classVal,1));
		if(result->second == false){
			result->first->second++;
		}
	}
	for(it=classCount.begin();it!=classCount.end();it++){
		counts.push_back(it->second);
	}
	infoGainOfData = entropy(counts);
}

// function to determine the splitting attribute
// attr: candidate attributes for splitting attribute, attr[i]=1 if already used
// data: data row nos(in the file and index in "fileContent" vector) used for calculating information gains
int select(vector <int> *attr,vector <int> data)
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
	selectedAttribute=select(&attr,data);
	root->attribute = selectedAttribute;

	map<int, vector <int> > dividedData;
	map<int, vector <int> >::iterator it;
	node childNode;
	int attrVal;

	for(i=0;i<fileContent.size();i++){
		attrVal = fileContent[i][selectedAttribute];
		vector <int> x;
		x.push_back(i);
		pair<pair<int, int>::iterator, bool> result =  dividedData.insert(make_pair(attrVal,x));
		if(result->second == false){
			result->first->second.push_back(i);
		}
	}
	for(i=0,it=dividedData.begin();it!=dividedData.end();it++,i++){
		childNode = create();
		root->child[i] = childNode;
		decision(&attr, it->second, root->child[i]);
	}

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
			// getCardinality();
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

	return 0;
}