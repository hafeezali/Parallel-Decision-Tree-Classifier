#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "math.h"
#include "limits.h"

#define INT_MIN -9999999
#define trainingData "myDataset.txt"
#define testingData "myDataset.txt"
#define M 6
#define N 14
#define trainFileData(row,col) trainFileData[row*M+col]
#define testFileData(row,col) testFileData[row*M+col]

using namespace std;

vector <vector <int> > trainFile;
vector <vector <int> > testFile;

int *d_trainFileData, *d_cardinality;

dim3 blocks(M-1);
dim3 threads(M,N);

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

__global__ void getInfoGains(int *attr,int *data,int dataSize,float *infoGains,int *trainFileData,int *cardinality)
{
	infoGains[blockIdx.x]=INT_MIN;
	__syncthreads();
	if(attr[blockIdx.x]==0){
		int tid,bid;
		float infoGain,intermediateGain;
		__shared__ int attrValCount[cardinality[bid]+1];
		__shared__ int classAttrValCount[cardinality[bid]+1][cardinality[M]+1];
		tid=threadIdx.x;
		bid=blockIdx.x;
		int classVal = trainFileData(data[tid],M-1);
		int attrVal = trainFileData(data[tid],bid);
		atomicAdd(&attrValCount[attrVal],1);
		atomicAdd(&classAttrValCount[attrVal][classVal],1);
		__syncthreads();
		infoGain=0;
		if(tid==0){
			int i,j;
			for(i=1;i<=cardinality[bid];i++){
				intermediateGain=0;
				for(j=1;j<=cardinality[M];j++){
					intermediateGain+=((float)classAttrValCount[i][j]/(float)attrValCount[i])*(log((float)classAttrValCount[i][j]/(float)attrValCount[i]));
				}
				intermediateGain*=(float(attrValCount[i])/(float)dataSize);
				infoGain-=intermediateGain;
			}
			infoGains[bid]=infoGain;
		}
	}
}

__global__ void getInfoGainOfData(int *data,int dataSize,int *trainFileData,int *cardinality,int *infoGainOfData)
{
	__shared__ classValCount[cardinality[M]+1];
	if(threadIdx.x==0){
		for(i=0;i<=cardinality[M];i++){
			classValCount[i]=0;
		}
	}
	__syncthreads();
	int classVal,i;
	classVal=trainFileData(data[threadIdx.x],M);
	atomicAdd(&classValCount[classVal],1);
	__syncthreads();
	float infoGain;
	infoGain=0;
	for(i=1;i<=cardinality[M];i++){
		infoGain+=((double)classValCount[i]/(double)dataSize)*(log((double)classValCount/(double)dataSize));
	}
	infoGainOfData=-1*infoGain;
}

int popularVote(int *data,int dataSize)
{
	int i,outputClass,ans,maxVal;
	map <int, int> dataCount;
	map <int, int>::iterator it;
	for(i=0;i<dataSize;i++){
		outputClass = trainFile[data[i]][M-1];
		if(dataCount.find(outputClass)==dataCount.end()){
			dataCount.insert(make_pair(outputClass,1));
		}
		else{
			dataCount[outputClass]++;
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

void decision(int *h_attr,int *h_data, node* root,int h_dataSize)
{
	int flag,h_selectedAttribute,i,maxGain;
	if(h_dataSize==0){
		return;
	}
	flag=1;
	for(i=1;i<h_dataSize;i++){
		if(trainFile[h_data[i]][M-1]!=trainFile[h_data[i-1]][M-1]){
			flag=0;
			break;
		}
	}
	if(flag==1){
		root->val=trainFile[h_data[0]][M-1];
		return;
	}

	int *d_attr, *d_data;
	float *d_infoGains,*d_infoGainOfData;
	float h_infoGains[M-1], h_infoGainOfData;

	cudaMalloc((void**)&d_attr,M*sizeof(int));
	cudaMalloc((void**)&d_data,h_dataSize*sizeof(int));
	cudaMalloc((void**)&d_infoGains,(M-1)*sizeof(float));
	cudaMemcpy((void*)d_attr,(void*)h_attr,M*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_data,(void*)h_data,N*sizeof(int),cudaMemcpyHostToDevice);

	getInfoGains<<<blocks,h_dataSize>>>(d_attr,d_data,h_dataSize,d_infoGains,d_trainFileData,d_cardinality);

	cudaMemcpy((void*)h_infoGains,(void*)d_infoGains,(M-1)*sizeof(float),cudaMemcpyDeviceToHost);

	cudaMalloc((void**)&d_infoGainOfData,sizeof(float));

	getInfoGainOfData<<<1,h_dataSize>>>(d_data,h_dataSize,d_trainFileData,d_cardinality,d_infoGainOfData);

	cudaMemcpy((void*)h_infoGainOfData,(void*)d_infoGainOfData,sizeof(float),cudaMemcpyDeviceToHost);

	maxGain=INT_MIN;
	h_selectedAttribute=-1;
	for(i=1;i<M-1;i++){
		h_infoGains[i]=h_infoGainOfData-h_infoGains[i];
		if(h_infoGains[i]>maxGain){
			maxGain=h_infoGains[i];
			h_selectedAttribute=i;
		}
	}

	root->attribute = h_selectedAttribute;

	if(h_selectedAttribute==-1){
		root->val = popularVote(h_data, h_dataSize);
		return;
	}

	map<int, vector <int> > dividedData;
	map<int, vector <int> >::iterator it;
	int attrVal;

	for(i=0;i<data.size();i++){
		attrVal = trainFile[data[i]][h_selectedAttribute];
		if(dividedData.find(attrVal) == dividedData.end()){
			vecotr<int> x;
			x.push_back(data[i]);
			dividedData.insert(make_pair(attrVal,x));
		}
		else{
			dividedData[attrVal].push_back(data[i]);
		}
	}
	for(i=0;it=dividedData.begin();it!=dividedData.end();it++,i++){
		root->numOfChildren++;
		node* childNode;
		childNode = create();
		childNode->branchVal = it->first;
		root->child[i] = childNode;

		decision(h_attr, it->second, childNode)
	}
}

__global__ void getCardinality(int *trainFileData,int *cardinality)
{
	__shared__ int x[10];
	int bid,tid;
	unsigned int i;
	bid=blockIdx.x;
	tid=threadIdx.x;
	if(tid<10){
		x[tid]=0;
	}
	__syncthreads();
	x[trainFileData(tid,bid)]==1;
	__syncthreads();
	for(i=1;i<10;i*=2){
		int index = 2*i*tid;
		if(index+i<10){
			x[index]+=x[index+i];
		}
		__syncthreads();
	}
	if(tid==0){
		cardinality[bid]=x[0];
	}
}

// function for printing and debugging decision tree : bfs traversal
void printDecisionTree(node *root)
{
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
	int i;
	node* root;

	readCSV("training");

	int h_trainFileData[N*M];

	for(i=0;i<N*M;i++){
		h_trainFileData[i] = trainFile[i/M][i%M];
	}

	int h_data[N],h_attr[M];

	for(i=0;i<N;i++){
		h_data[i]]=i;
	}

	for(i=0;i<M;i++){
		h_attr[i]=0;
	}

	cudaMalloc((void**)&d_trainFileData,N*M*sizeof(int));
	cudaMemcpy((void*)d_trainFileData,(void*)h_trainFileData,M*N*sizeof(int),cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_cardinality,M*sizeof(int));
	cudaMemset(d_cardinality,0,M*sizeof(int));
	getCardinality<<<M,threads>>>(d_trainFileData,d_cardinality);

	root = create();
	decision(h_attr,h_data,root,N);

	//print decision tree
	//printDecisionTree(root);

	// test decision tree
	test(root);

	return 0;
}