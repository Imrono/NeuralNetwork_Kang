#include "TrainNeuralNetwork.h"
#include <assert.h>

double TrainNeuralNetwork::getMSE(const vector<double>& actualOutput, const vector<double>& targetOutput) 
{
	size_t count = actualOutput.size();
	double dMSE = 0.0f;
	for (unsigned i = 0; i < count; ++i)
		dMSE += (actualOutput[i]-targetOutput[i]) * (actualOutput[i]-targetOutput[i]);
	dMSE /= 2.0f;
	return dMSE;
}

double TrainNeuralNetwork::getMSE_AVG()
{
	size_t numSample = samples.size();
	double dMSE = 0.0f;
	for (unsigned i = 0; i < numSample; i++) {
		dMSE += getMSE(actualOutput[i], samples[i].y);
	}
	dMSE /= 2.0f;
	dMSE /= numSample;
	return dMSE;
}

void TrainNeuralNetwork::Train(NeuralNetwork& NN, const vector<Data>* const inData)
{
	if (inData)
		samples = *inData;

	size_t xSize = samples[0].x.size();
	size_t ySize = samples[0].y.size();
	size_t sampleSize = samples.size();
	assert(sampleSize > 0);
	assert(xSize == NN.m_Layers[0]->m_Neurons.size());
	assert(ySize == NN.m_Layers[NN.m_Layers.size()-1]->m_Neurons.size());

	unsigned count = 0;
	double *tmpActualOutput = new double[ySize];
	do {
		actualOutput.clear();
		for (int sampleIdx = 0; sampleIdx < sampleSize; sampleIdx++) {
			vector<double> tmpY;
			NN.forwardCalc_NN(samples[sampleIdx].x, tmpY);
			NN.BackPropagate_NN(&(tmpY[0]), &(samples[sampleIdx].y[0]), ySize);

			actualOutput.push_back(tmpY);
		}
		count++;
	} while (getMSE_AVG() < residualErr || count >= numIter);
	delete tmpActualOutput;
}