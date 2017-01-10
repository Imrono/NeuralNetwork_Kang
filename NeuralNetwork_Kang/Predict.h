#ifndef _PREDICT_H_
#define _PREDICT_H_

#include "NeuralNetwork.h"
#include "TrainNeuralNetwork.h"
#include <vector>
using namespace std;

class PredictNN
{
public:
	void myPredict(NeuralNetwork& NN, Data& actual);
};

#endif
