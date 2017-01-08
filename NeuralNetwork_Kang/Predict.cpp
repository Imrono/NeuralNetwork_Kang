#include "Predict.h"

void PredictNN::myPredict(NeuralNetwork& NN, Data& actual)
{
	NN.forwardCalc_NN(actual.x, actual.y);
}