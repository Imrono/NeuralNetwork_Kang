#include "TrainNeuralNetwork.h"
#include <assert.h>
#include <cstdio>

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
		dMSE += getMSE(m_actualOutput[i], samples[i].y);
	}
	dMSE /= numSample;
	return dMSE;
}

void TrainNeuralNetwork::Train(NeuralNetwork& NN)
{
	size_t xSize = samples[0].x.size();
	size_t ySize = samples[0].y.size();
	size_t sampleSize = samples.size();
	assert(sampleSize > 0);
	assert(xSize == NN.m_Layers[0]->m_Neurons.size());
	assert(ySize == NN.m_Layers[NN.m_Layers.size()-1]->m_Neurons.size());

	unsigned count = 0;
	double tmpMSE = 0.0f;
	double *tmpActualOutput = new double[ySize];
	do {
		if (m_actualOutput.size() > 0)
			m_actualOutput[count].clear();
		for (int sampleIdx = 0; sampleIdx < sampleSize; sampleIdx++) {
			vector<double> tmpY;
			NN.forwardCalc_NN(samples[sampleIdx].x, tmpY);
			NN.BackPropagate_NN(&(tmpY[0]), &(samples[sampleIdx].y[0]), ySize);

			m_actualOutput.push_back(tmpY);
		}
		tmpMSE = getMSE_AVG();
		printf("== ITERATION %4d(%4d): with MSE %.6f (%.6f) ==\n", count, numIter, tmpMSE, residualErr);
		count++;
	} while (tmpMSE > residualErr && count < numIter);
	delete tmpActualOutput;
	
	NN.weightToString();
}

void TrainNeuralNetwork::samplesToString()
{
	size_t sampleSize = samples.size();
	printf("sample number: %3d\n", sampleSize);
	for(size_t i = 0; i < sampleSize; i++) {
		size_t xSize = samples[i].x.size();
		size_t ySize = samples[i].y.size();
		printf("**%3d** x: ", i);
		for(size_t xi = 0; xi < xSize; xi++) {
			printf("%.5f ", samples[i].x[xi]);
		}
		printf("\t y: ", i);
		for(size_t yi = 0; yi < ySize; yi++) {
			printf("%.5f ", samples[i].y[yi]);
		}
		printf("\n", i);
	}
}