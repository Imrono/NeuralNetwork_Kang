#ifndef _TRAIN_NEURALNETWORK_H_
#define _TRAIN_NEURALNETWORK_H_

#include "NeuralNetwork.h"
#include <vector>
using namespace std;

struct Data  
{  
    vector<double> x;       //输入数据  
    vector<double> y;       //输出数据  
}; 

class TrainNeuralNetwork
{
public:
	TrainNeuralNetwork(double inResidualErr = 0.0f, unsigned inNumIter = 0) : residualErr(inResidualErr), numIter(inNumIter) {}
	~TrainNeuralNetwork() {}

	void setSamples(const vector<Data>& inData);
	void setNumIter(const unsigned inNumIter) { numIter = inNumIter;}
	void setResidualErr(const double inResidualErr) { residualErr = inResidualErr;}

	double getMSE(const vector<double>& actualOutput, const vector<double>& targetOutput);
	double getMSE_AVG();
	
	void Train(NeuralNetwork& NN);
	
	
private:
	double residualErr;
	unsigned numIter;
	vector<Data> samples;
	vector<vector<double>> m_actualOutput;
	
	void samplesToString();
};

#endif