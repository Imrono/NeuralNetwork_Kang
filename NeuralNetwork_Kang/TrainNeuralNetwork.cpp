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
		double tmpMSE = getMSE(m_actualOutput[i], samples[i].y);
		dMSE += tmpMSE;
		//printf("Sample:%d,tmpMSE:%f actOut:", i, tmpMSE);
		//for (size_t j = 0; j < m_actualOutput[i].size(); j++) {
		//	printf("%f ", m_actualOutput[i][j]);
		//}
		//printf("Out:");
		//or (size_t j = 0; j < samples[i].y.size(); j++) {
		//	printf("%f ", samples[i].y[j]);
		//}
		//printf("\n");
	}
	//printf("\n");
	dMSE /= numSample;
	return dMSE;
}

void TrainNeuralNetwork::setSamples(const vector<Data>& inData) 
{ 
	samples = inData; 
	m_actualOutput.reserve(samples.size()); 
	samplesToString();
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
	double tmpLastMSE = 0.0f;
	double *tmpActualOutput = new double[ySize];
	do {
		tmpLastMSE = tmpMSE;
		if (m_actualOutput.size() > 0)
			m_actualOutput.clear();
		for (int sampleIdx = 0; sampleIdx < sampleSize; sampleIdx++) {
			vector<double> tmpY;
			NN.forwardCalc_NN(samples[sampleIdx].x, tmpY);
			NN.BackPropagate_NN(&(tmpY[0]), &(samples[sampleIdx].y[0]), ySize);
			m_actualOutput.push_back(tmpY);
		}
		tmpMSE = getMSE_AVG();
		if (!(count%500))
			printf("== ITERATION %4d(%4d): with MSE %.12f (%.12f) dMSE:%.12f outputSize:%d ==\n", count, numIter, tmpMSE, residualErr, tmpLastMSE-tmpMSE, m_actualOutput.size());
		count++;
	} while (tmpMSE > residualErr && count < numIter);
	printf("**** TOTAL ITERATION %4d(%4d): with MSE %.12f (%.12f) outputSize:%d ****\n", count, numIter, tmpMSE, residualErr, m_actualOutput.size());
	delete tmpActualOutput;
	
	//NN.weightToString();
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