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
private:
	unsigned numIter;
	vector<Data> samples;
};

#endif