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
	void setSamples(const vector<Data>& inData) { samples = inData;}
private:
	unsigned numIter;
	vector<Data> samples;
};

#endif