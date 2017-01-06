#ifndef _TRAIN_NEURALNETWORK_H_
#define _TRAIN_NEURALNETWORK_H_

#include "NeuralNetwork.h"
#include <vector>
using namespace std;

struct Data  
{  
    vector<double> x;       //��������  
    vector<double> y;       //�������  
}; 

class TrainNeuralNetwork
{
public:
	void setSamples(const vector<Data>& inData) { samples = inData;}
	double getMSE(const vector<double>& actualOutput, const vector<double>& targetOutput);
	double getMSE_AVG();
	
	void Train(); 
	
	
private:
	double residualErr;
	unsigned numIter;
	vector<Data> samples;
	vector<vector<double>> actualOutput;
};

#endif