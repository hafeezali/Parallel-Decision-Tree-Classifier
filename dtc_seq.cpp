#include <vector>
#include <map>
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
double infoGainOfData;

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
	for(i=1;i<fileContent[0].size();i++){
		set <string> values;
		for(j=0;j<fileContent.size();j++){
			values.insert(fileContent[j][i]);
		}
		cardinal[i]=set.size();
	}
	cardinality=cardinal;
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
		entropy += ((counts[i]/total)*(log(counts[i]/total)/log(2)));
	}
	return entropy;
}

double infoGain(int attr,vector <int> data)
{
	int i,branchVal;
	map<int, int> branchCount;
	map<int, vector<int>> dataElements;
	for(i=0;i<data.size();i++){
		branchVal = fileContent[i][attr];
		pair<pair<int, int>::iterator, bool> result = branchCount.insert(make_pair(branchVal,1));
		if(result.second == false){
			result.first->second++;
		}
		vector <int> x;
		x.push_back(i);
		pair<pair<int, int>::iterator, bool> result = dataElements.insert(make_pair(branchVal,x));
		if(result.second == false){
			result.first->second.push_back(i);
		}
	}
	
}

void getInfoGainOfData()
{
	int i,classVal;
	map<int, int> classCount;
	map<int, int>::iterator it;
	vector<int> counts;
	for(i=0;i<fileContent.size();i++){
		classVal = fileContent[i][numOfAttrib-1];
		pair<pair<int, int>::iterator, bool> result =  classCount.insert(make_pair(classVal,1));
		if(result.second == false){
			result.first->second++;
		}
	}
	for(it=classCount.begin();it!=classCount.end();it++){
		counts.push_back(it->second);
	}
	infoGainOfData = entropy(counts);
}

int select(vector <int> *attr,vector <int> data)
{
	int i,splitAttr;
	double iGain,maxIGain;
	maxIGain = INT_MIN;
	for(i=1;i<attr.size()-1;i++){
		if(attr[i]==0){
			iGain = infoGain(i,data);
			if(iGain>maxIGain){
				maxIGain = iGain;
				splitAttr = i;
			}
		}
	}
	if(maxIGain==INT_MIN){
		return -1;
	}
	attr[splitAttr]=1;
	return splitAttr;
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
	selectedAttribute=select(&attr,data);
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
	getInfoGainOfData();

	for(i=0;i<=numOfDataEle;i++){
		data.push_back(i);
	}
	for(i=0;i<=numOfAttrib+2;i++){
		attr.push_back(0);
	}

	root = create();
	decision(attr,data,root);

	return 0;
}