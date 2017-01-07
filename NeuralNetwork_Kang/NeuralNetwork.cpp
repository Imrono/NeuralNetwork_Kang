// NeuralNetwork.cpp: implementation of the NeuralNetwork class.
//
//////////////////////////////////////////////////////////////////////
#include "NeuralNetwork.h"
#include <cstdlib>
#include <assert.h>

///////////////////////////////////////////////////////////////////////
//  NeuralNetwork class definition
///////////////////////////////////////////////////////////////////////
NeuralNetwork::~NeuralNetwork()
{
	reset();
}

void NeuralNetwork::reset()
{
	for(auto layer_It = m_Layers.begin(); layer_It < m_Layers.end(); layer_It++)
		delete *layer_It;
	m_Layers.clear();
}

void NeuralNetwork::create(const unsigned numLayers, const unsigned* const ar_nodes)
{
	reset();
	nLayer = numLayers;
	for (unsigned i = 0; i < numLayers; i++) {
		nodes.push_back(ar_nodes[i]);
	}
	
	for (unsigned i = 0; i < numLayers; i++) { 
        NNLayer* pLayer = new NNLayer(); 
        if (i == 0) {
			pLayer->m_pPrevLayer = nullptr;
            pLayer->addNeurals(nodes[i], 0); 
        } else { 
            pLayer->m_pPrevLayer = m_Layers[i-1]; 
            pLayer->addNeurals(nodes[i], nodes[i-1]); 
			
            unsigned numWeights = nodes[i] * (nodes[i-1]+1); 		// 有一个是bias 
            for (unsigned k = 0; k < numWeights; k++) { 
                pLayer->m_Weights.push_back(0.05*rand()/RAND_MAX); 	// 初始化权重在0~0.05 
            } 
        } 
        m_Layers.push_back(pLayer); 
    }
}

///////////////////////////////////////////////////////////////////////
//  NNLayer class definition
///////////////////////////////////////////////////////////////////////
void NNLayer::addNeurals(unsigned numNeural, unsigned prevNumNeural) 
{ 
    for (unsigned i = 0; i < numNeural; i++) { 
        NNNeuron neuron; 
        neuron.output = 0; 
        for (unsigned k = 0; k < prevNumNeural; k++) { 
            NNConnection connection; 
            connection.WeightIndex = i*prevNumNeural + k;	// 设置权重索引 
            connection.NeuronIndex = k;    					// 设置前层结点索引 
            neuron.m_Connections.push_back(connection); 
        } 
        m_Neurons.push_back(neuron); 
    }
	m_Neurons.push_back(NNNeuron(1.0f));					// bias 
}

///////////////////////////////////////////////////////////////////////
//  forwardCalc
///////////////////////////////////////////////////////////////////////
void NeuralNetwork::forwardCalc_NN(const double* const inputVector, const unsigned iCount, 
								   double* const outputVector /* =NULL */, const unsigned oCount /* =0 */)
{
	auto layer_It = m_Layers.begin();
	// first layer is inputVector
	if  (layer_It < m_Layers.end()) {
		auto neuron_It = (*layer_It)->m_Neurons.begin();
		unsigned count = 0;
		assert(iCount == (*layer_It)->m_Neurons.size());  // there should be exactly one neuron per input
		while((neuron_It < (*layer_It)->m_Neurons.end()) && (count < iCount))
			(neuron_It++)->output = inputVector[count++];
	}
	layer_It++;
	
	for(; layer_It < m_Layers.end(); layer_It++)
		(*layer_It)->forwardCalc_Layer();
	
	// load up output vector with results
	if (outputVector) {
		layer_It = m_Layers.end() - 1;
		unsigned i = 0;
		for (auto neuron_It = (*layer_It)->m_Neurons.begin(); i < oCount; neuron_It++) {
			outputVector[i] = neuron_It->output;
			i++;
		}
	}
}

void NNLayer::forwardCalc_Layer()
{
	assert(m_pPrevLayer);
	for(auto neuron_It = m_Neurons.begin(); neuron_It < m_Neurons.end(); neuron_It++) {
		double dSum = 0.0f;
		for (auto conn_It = neuron_It->m_Connections.begin(); conn_It < neuron_It->m_Connections.end(); conn_It++) {
			assert(conn_It->WeightIndex < m_Weights.size());
			assert(conn_It->NeuronIndex < m_pPrevLayer->m_Neurons.size());
			dSum += m_Weights[conn_It->WeightIndex] * m_pPrevLayer->m_Neurons[conn_It->NeuronIndex].output;
		}
		neuron_It->output = SIGMOID(dSum);
	}
}

///////////////////////////////////////////////////////////////////////
//  BackPropagate
///////////////////////////////////////////////////////////////////////
void NeuralNetwork::BackPropagate_NN(const double* const actualOutput, const double* const desiredOutput, const unsigned count)
{
	// backpropagates through the neural net
	assert((actualOutput) && (desiredOutput) && (count < 256));
	assert(m_Layers.size() >= 2);  // there must be at least two layers in the net
	
	if ((!actualOutput) || (!desiredOutput) || (count >= 256))
		return;
	
	// Err is output error of the entire neural net
	// Xn is the output vector on the n-th layer
	// Xnm1 is the output vector of the previous layer

	unsigned numOutput = (*(m_Layers.end()-1))->m_Neurons.size();
	vector<double> dErr_dXlast(numOutput);
	// start the process by calculating dErr_dXn for the last layer.
	// for the standard MSE Err function (i.e., 0.5*sumof( (actual-target)^2 ), this differential is simply
	for (unsigned i = 0; i < numOutput; ++i)
		dErr_dXlast[i] = actualOutput[i] - desiredOutput[i];
	
	vector<vector<double>> differentials;
	unsigned iSize = m_Layers.size();
	differentials.resize(iSize);
	// store Xlast and reserve memory for the remaining vectors stored in differentials
	for (unsigned i = 0; i < iSize-1; ++i) {
		differentials[i].resize(m_Layers[i]->m_Neurons.size(), 0.0);
	}
	differentials[iSize-1] = dErr_dXlast;  // last one
	
	// now iterate through all layers including the last but excluding the first, and ask each of
	// them to backpropagate error and adjust their weights, and to return the differential
	// dErr_dXnm1 for use as the input value of dErr_dXn for the next iterated layer
	unsigned i = iSize - 1;
	for (auto layer_It = m_Layers.end() - 1; layer_It > m_Layers.begin(); layer_It--) {
		(*layer_It)->BackPropagate_Layer(differentials[i], differentials[i-1], m_etaLearningRate);
		i--;
	}
}

void NNLayer::BackPropagate_Layer(vector<double>& dErr_dXn   /* in */, 
							vector<double>& dErr_dXnm1 /* out */,
							const double eta)
{
	// Err is output error of the entire neural net
	// Xn is the output vector on the n-th layer
	// Xnm1 is the output vector of the previous layer

	assert(dErr_dXn.size() == m_Neurons.size());
	assert(m_pPrevLayer);
	assert(dErr_dXnm1.size() == m_pPrevLayer->m_Neurons.size());
	
	vector<double> dErr_dYn(m_Neurons.size());

	vector<double> dErr_dWn(m_Weights.size(), 0.0);	// important to initialize to zero
// 	double* dErr_dWn = (double*)(alloca( sizeof(double) *  m_Weights.size())); // alloca in stack
	for (unsigned i = 0; i < m_Weights.size(); ++i)
		dErr_dWn[i] =0.0;
	
	// 1. calculate dErr_dYn = F'(Yn) * dErr_Xn
	for (unsigned i = 0; i < m_Neurons.size(); ++i) {
		assert(i < dErr_dYn.size());
		assert(i < dErr_dXn.size());
		dErr_dYn[i] = DSIGMOID(m_Neurons[i].output) * dErr_dXn[i];
	}
	
	// 2. calculate dErr_Wn = Xnm1 * dErr_Yn
	unsigned i = 0;
	for (auto neuron_It = m_Neurons.begin(); neuron_It < m_Neurons.end(); neuron_It++) {
		for (auto conn_It = neuron_It->m_Connections.begin(); conn_It < neuron_It->m_Connections.end(); conn_It++) {
			assert(conn_It->NeuronIndex < m_pPrevLayer->m_Neurons.size());
			assert(conn_It->WeightIndex < m_Weights.size()); // m_Weights.size() == dErr_dWn.size()
			assert(i < dErr_dYn.size());
			dErr_dWn[conn_It->WeightIndex] += dErr_dYn[i] * m_pPrevLayer->m_Neurons[conn_It->NeuronIndex].output;
		}
		i++;
	}
	
	// * calculate dErr_dXnm1 = Wn * dErr_dYn, which is needed as the input value of
	i = 0;
	for (auto neuron_It = m_Neurons.begin(); neuron_It < m_Neurons.end(); neuron_It++) {
		for (auto conn_It = neuron_It->m_Connections.begin(); conn_It < neuron_It->m_Connections.end(); conn_It++) {
			assert(conn_It->NeuronIndex < dErr_dXnm1.size());
			assert(conn_It->WeightIndex < m_Weights.size());
			assert(i < dErr_dYn.size());	
			dErr_dXnm1[conn_It->NeuronIndex] += dErr_dYn[i] * m_Weights[conn_It->WeightIndex];
		}
		i++;
	}
	
	// 3. update the weights of this layer neuron using dErr_dW and the learning rate eta
	for (unsigned j = 0; j < m_Weights.size(); ++j) {
		m_Weights[j] -= eta * dErr_dWn[j];
	}
}
