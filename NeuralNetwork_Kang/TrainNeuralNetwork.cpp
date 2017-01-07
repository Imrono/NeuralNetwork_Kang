#include "TrainNeuralNetwork.h"

double TrainNeuralNetwork::getMSE(const vector<double>& actualOutput, const vector<double>& targetOutput) 
{
	unsigned count = actualOutput.size();
	double dMSE = 0.0f;
	for (unsigned i = 0; i < count; ++i)
		dMSE += (actualOutput[i]-targetOutput[i]) * (actualOutput[i]-targetOutput[i]);
	dMSE /= 2.0f;
	return dMSE;
}

double TrainNeuralNetwork::getMSE_AVG()
{
	unsigned numSample = samples.size();
	double dMSE = 0.0f;
	for (unsigned i = 0; i < numSample; i++) {
		dMSE += getMSE(actualOutput[i], samples[i].y);
	}
	dMSE /= 2.0f;
	dMSE /= numSample;
	return dMSE;
}