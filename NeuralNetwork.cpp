// NeuralNetwork.cpp: implementation of the NeuralNetwork class.
//
//////////////////////////////////////////////////////////////////////
#include "NeuralNetwork.h"
#include <malloc.h>  // for the _alloca function

///////////////////////////////////////////////////////////////////////
//  NeuralNetwork class definition
///////////////////////////////////////////////////////////////////////
NeuralNetwork::~NeuralNetwork()
{
	for(vector<NNLayer*>::iterator it = m_Layers.begin(); it < m_Layers.end(); it++)
		delete *it;
	m_Layers.clear();
}

void create(unsigned numLayers, unsigned* ar_nodes)
{
	nLayer = numLayers;
	for (int i = 0; i < numLayers; i++) {
		nodes.push_back(ar_nodes[i]);
	}
	initializeNetwork();
}

void NeuralNetwork::initializeNetwork() 
{ 
    for (unsigned i = 0; i < nLayer; i++) { 
        NNlayer* pLayer = new NNlayer(); 
        if (i == 0) {
			pLayer->preLayer = nullptr;
            pLayer->addNeurals(nodes[i], 0); 
        } else { 
            pLayer->preLayer = m_layers[i-1]; 
            pLayer->addNeurals(nodes[i], nodes[i-1]); 
			
            unsigned numWeights = nodes[i] * (nodes[i-1]+1); // 有一个是bias 
            for (unsigned k = 0; k < numWeights; k++) { 
                pLayer->m_weights.push_back(0.05*rand()/RAND_MAX); 	// 初始化权重在0~0.05 
            } 
        } 
        m_layers.push_back(pLayer); 
    } 
}
///////////////////////////////////////////////////////////////////////
//  NNLayer class definition
///////////////////////////////////////////////////////////////////////
NNLayer::~NNLayer()
{
	for(vector<NNNeuron*>::iterator nit = m_Neurons.begin(); nit < m_Neurons.end(); nit++)
		delete *nit;
	m_Neurons.clear();
	for(vector<NNWeight*>::iterator wit = m_Weights.begin(); wit < m_Weights.end(); wit++)
		delete *wit;
	m_Weights.clear();
}

void NNlayer::addNeurals(unsigned numNeural, unsigned prevNumNeural) 
{ 
    for (unsigned i = 0; i < numNeural; i++) { 
        NNneural neural; 
        neural.output = 0; 
        for (unsigned k = 0; k < prevNumNeural; k++) { 
            NNconnection connection; 
            connection.weightIdx = i*prevNumNeural + k; // 设置权重索引 
            connection.neuralIdx = k;    					 // 设置前层结点索引 
            neural.m_connection.push_back(connection); 
        } 
        m_neurals.push_back(neural); 
    }
	m_neurals.push_back(NNNeuron(1.0f));					 // bias 
}

///////////////////////////////////////////////////////////////////////
//  NNNeuron
///////////////////////////////////////////////////////////////////////
void NNNeuron::AddConnection(unsigned iNeuron, unsigned iWeight)
{
	m_Connections.push_back(NNConnection(iNeuron, iWeight));
}

void NNNeuron::AddConnection(const NNConnection& conn)
{
	m_Connections.push_back(conn);
}

///////////////////////////////////////////////////////////////////////
//  forwardCalc
///////////////////////////////////////////////////////////////////////
void NeuralNetwork::forwardCalc_NN(const double const* inputVector, const unsigned iCount, 
								   const double* outputVector /* =NULL */, const unsigned oCount /* =0 */)
{
	auto layer_It = m_Layers.begin();
	// first layer is inputVector
	if  (layer_It < m_Layers.end()) {
		auto neuron_It = (*layer_It)->m_Neurons.begin();
		int count = 0;
		ASSERT(iCount == (*layer_It)->m_Neurons.size());  // there should be exactly one neuron per input
		while((neuron_It < (*layer_It)->m_Neurons.end()) && (count < iCount))
			(*(neuron_It++))->output = inputVector[count++];
	}
	layer_It++;
	
	for(; layer_It < m_Layers.end(); layer_It++)
		(*layer_It)->forwardCalc_Layer();
	
	// load up output vector with results
	if (outputVector) {
		layer_It = m_Layers.end() - 1;
		auto neuron_It = (*layer_It)->m_Neurons.begin();
		for (int i = 0; i < oCount; ++i) {
			outputVector[i] = (*neuron_It)->output;
			neuron_It++;
		}
	}
}

void NNLayer::forwardCalc_Layer()
{
	ASSERT(m_pPrevLayer);
	for(auto neuron_It = m_Neurons.begin(); neuron_It < m_Neurons.end(); neuron_It++) {
		double dSum = 0.0f;
		for (auto conn_It = neuron_It->m_Connections.begin(); conn_It < neuron_It->m_Connections.end(); conn_It++) {
			ASSERT(conn_It->WeightIndex < m_Weights.size());
			ASSERT(conn_It->NeuronIndex < m_pPrevLayer->m_Neurons.size());
			dSum += m_Weights[conn_It->WeightIndex] * m_pPrevLayer->m_Neurons[conn_It->NeuronIndex]->output;
		}
		neuron_It->output = SIGMOID(dSum);
	}
}

///////////////////////////////////////////////////////////////////////
//  BackPropagate
///////////////////////////////////////////////////////////////////////
void NeuralNetwork::Backpropagate(double *actualOutput, double *desiredOutput, unsigned count)
{
	// backpropagates through the neural net
	ASSERT((actualOutput) && (desiredOutput) && (count < 256));
	ASSERT(m_Layers.size() >= 2);  // there must be at least two layers in the net
	
	if ((!actualOutput) || (!desiredOutput) || (count >= 256))
		return;
	
	// proceed from the last layer to the first, iteratively
	// We calculate the last layer separately, and first, since it provides the needed derviative
	// (i.e., dErr_dXnm1) for the previous layers
	
	// Err is output error of the entire neural net
	// Xn is the output vector on the n-th layer
	// Xnm1 is the output vector of the previous layer

	auto layer_It = m_Layers.end() - 1;
	vector<double> dErr_dXlast((*layer_It)->m_Neurons.size());
	// start the process by calculating dErr_dXn for the last layer.
	// for the standard MSE Err function (i.e., 0.5*sumof( (actual-target)^2 ), this differential is simply
	for (unsigned i = 0; i < (*layer_It)->m_Neurons.size(); ++i)
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
	for (unsigned i = iSize - 1, layer_It; layer_It > m_Layers.begin(); i--, layer_It--)
		(*layer_It)->Backpropagate(differentials[i], differentials[i-1], m_etaLearningRate);
}

void NNLayer::Backpropagate(vector<double>& dErr_dXn   /* in */, 
							vector<double>& dErr_dXnm1 /* out */,
							double etaLearningRate )
{
	// nomenclature (repeated from NeuralNetwork class):
	//
	// Err is output error of the entire neural net
	// Xn is the output vector on the n-th layer
	// Xnm1 is the output vector of the previous layer

	ASSERT(dErr_dXn.size() == m_Neurons.size());
	ASSERT(m_pPrevLayer);
	ASSERT(dErr_dXnm1.size() == m_pPrevLayer->m_Neurons.size());
	
	vector<double> dErr_dYn(m_Neurons.size());

// 	vector< double > dErr_dWn( m_Weights.size(), 0.0 );  // important to initialize to zero
	double* dErr_dWn = (double*)(_alloca( sizeof(double) *  m_Weights.size())); // alloca in stack
	for (int i = 0; i < m_Weights.size(); ++i)
		dErr_dWn[i] =0.0;
	
	// 1. calculate dErr_dYn = F'(Yn) * dErr_Xn
	for (unsigned i = 0; i < m_Neurons.size(); ++i) {
		ASSERT(i < dErr_dYn.size());
		ASSERT(i < dErr_dXn.size());
		dErr_dYn[i] = DSIGMOID(m_Neurons[i]->output) * dErr_dXn[i];
	}
	
	// 2. calculate dErr_Wn = Xnm1 * dErr_Yn
	for (unsigned i = 0, auto neuron_It = m_Neurons.begin(); neuron_It < m_Neurons.end(); i++, neuron_It++) {
		for (auto conn_It = (*neuron_It)->m_Connections.begin(); conn_It < (*neuron_It)->m_Connections.end(); conn_It++) {
			ASSERT((*conn_It)->NeuronIndex < m_pPrevLayer->m_Neurons.size());
			ASSERT((*conn_It)->WeightIndex < m_Weights.size()); // m_Weights.size() == dErr_dWn.size()
			ASSERT(i < dErr_dYn.size());
			dErr_dWn[(*conn_It)->WeightIndex] += dErr_dYn[i] * m_pPrevLayer->m_Neurons[(*conn_It)->NeuronIndex]->output;
		}
	}
	
	// * calculate dErr_dXnm1 = Wn * dErr_dYn, which is needed as the input value of
	for (unsigned i = 0, auto neuron_It = m_Neurons.begin(); neuron_It < m_Neurons.end(); , i++, neuron_It++) {
		for (auto conn_It = (*neuron_It)->m_Connections.begin(); conn_It < (*neuron_It)->m_Connections.end(); conn_It++) {
			ASSERT((*conn_It)->NeuronIndex < dErr_dXnm1.size());
			ASSERT((*conn_It)->WeightIndex < m_Weights.size());
			ASSERT(i < dErr_dYn.size());	
			dErr_dXnm1[(*conn_It).NeuronIndex] += dErr_dYn[i] * m_Weights[(*conn_It)->WeightIndex]->value;
		}
	}
	
	// 3. update the weights of this layer neuron using dErr_dW and the learning rate eta
	for (unsigned j = 0; j < m_Weights.size(); ++j) {
		m_Weights[j]->value -= etaLearningRate * dErr_dWn[j];
	}
}
