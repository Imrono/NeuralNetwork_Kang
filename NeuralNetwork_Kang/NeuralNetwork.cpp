// NeuralNetwork.cpp: implementation of the NeuralNetwork class.
//
//////////////////////////////////////////////////////////////////////
#include "NeuralNetwork.h"
#include <cstdio>
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
	m_nodes.clear();
}

///////////////////////////////////////////////////////////////////////
//  INITIAL NN
///////////////////////////////////////////////////////////////////////
void NeuralNetwork::create(const vector<size_t>& ar_nodes)
{
	reset();
	m_nLayer = ar_nodes.size();
	for (size_t i = 0; i < m_nLayer; i++) {
		m_nodes.push_back(ar_nodes[i]);
	}
	
	for (size_t i = 0; i < m_nLayer; i++) {
		bool isInOut = (i == 0 || i == m_nLayer - 1);
        NNLayer* pLayer = new NNLayer(i); 
        if (i == 0) {
			pLayer->m_pPrevLayer = nullptr;
            pLayer->addNeurals(m_nodes[i], 0, isInOut); 
        } else { 
            pLayer->m_pPrevLayer = m_Layers[i-1];
            pLayer->addNeurals(m_nodes[i], m_nodes[i-1], isInOut); 
			
			size_t numWeights = 0;
			if (1 == i)
				numWeights = m_nodes[i] * m_nodes[i-1];
			else
				numWeights = m_nodes[i] * (m_nodes[i-1]+1); // 有一个是bias 
			
            for (size_t k = 0; k < numWeights; k++) {
                pLayer->m_Weights.push_back(WEIGTH_INIT);
            } 
        } 
        m_Layers.push_back(pLayer); 
    }
	neuronLayerToString();
}

void NNLayer::addNeurals(size_t numNeural, size_t prevNumNeural, bool isInOut)
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
	if (!isInOut)
		m_Neurons.push_back(NNNeuron(1.0f));					// bias 
}

void NeuralNetwork::neuronLayerToString()
{
	for (size_t i = 0; i < m_nLayer; i++) {
		printf("Layer %3d in NN, with Neurons%3d (input:%3d)\n", i, m_Layers[i]->m_Neurons.size(), m_nodes[i]);
		m_Layers[i]->neuronToString();
	}
}

void NNLayer::neuronToString()
{
	size_t nNeuron = m_Neurons.size();
	for (size_t i = 0; i < nNeuron; i++) {
		//printf("\t\n");
	}
}

void NeuralNetwork::weightToString()
{
	printf("NeuralNetwork weights\n");
	for (size_t i = 0; i < m_nLayer; i++) {
		m_Layers[i]->weightToString();
	}
}

void NNLayer::weightToString()
{
	size_t weightSize = m_Weights.size();
	size_t prev_i = 0;
	size_t curr_i = 0;
	printf("Layer %3d, weightSize:%3d | ", layerId, weightSize);
	for(size_t wi = 0; wi < weightSize; wi++) {
		size_t prev_num = 0;
		if (m_pPrevLayer)
			prev_num = m_pPrevLayer->m_Neurons.size();
		else
			prev_num = 0;
		printf("W%d%d:%f ", curr_i, prev_i, m_Weights[wi]);
		prev_i = (++prev_i) % prev_num;
		curr_i = (++curr_i) / prev_num;
	}
	printf("\n");
}
///////////////////////////////////////////////////////////////////////
//  forwardCalc
///////////////////////////////////////////////////////////////////////
void NeuralNetwork::forwardCalc_NN(const vector<double>& inputVector, vector<double>& outputVector)
{
	auto layer_It = m_Layers.begin();
	// first layer is inputVector
	if  (layer_It < m_Layers.end()) {
		auto neuron_It = (*layer_It)->m_Neurons.begin();
		unsigned count = 0;
		size_t xSize = inputVector.size();
		assert(xSize == (*layer_It)->m_Neurons.size());  // there should be exactly one neuron per input
		while((neuron_It < (*layer_It)->m_Neurons.end()) && (count < xSize))
			(neuron_It++)->output = inputVector[count++];
		
	}
	layer_It++;
	
	// following layers
	for(; layer_It < m_Layers.end(); layer_It++)
		(*layer_It)->forwardCalc_Layer();
	
	// load up output vector with results
	outputVector.clear();
	layer_It = m_Layers.end() - 1;
	for (auto neuron_It = (*layer_It)->m_Neurons.begin(); neuron_It < (*layer_It)->m_Neurons.end(); neuron_It++)
		outputVector.push_back(neuron_It->output);
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
void NeuralNetwork::BackPropagate_NN(const double* const actualOutput, const double* const desiredOutput, size_t count)
{
	// backpropagates through the neural net
	assert((actualOutput) && (desiredOutput) && (count < 256));
	assert(m_Layers.size() >= 2);  // there must be at least two layers in the net
	
	if ((!actualOutput) || (!desiredOutput) || (count >= 256))
		return;
	
	// Err is output error of the entire neural net
	// Xn is the output vector on the n-th layer
	// Xnm1 is the output vector of the previous layer

	size_t numOutput = (*(m_Layers.end()-1))->m_Neurons.size();
	vector<double> dErr_dXlast(numOutput);
	// start the process by calculating dErr_dXn for the last layer.
	// for the standard MSE Err function (i.e., 0.5*sumof( (actual-target)^2 ), this differential is simply
	for (size_t i = 0; i < numOutput; ++i)
		dErr_dXlast[i] = actualOutput[i] - desiredOutput[i];
	
	vector<vector<double>> differentials;
	size_t iSize = m_Layers.size();
	differentials.resize(iSize);
	// store Xlast and reserve memory for the remaining vectors stored in differentials
	for (size_t i = 0; i < iSize-1; ++i) {
		differentials[i].resize(m_Layers[i]->m_Neurons.size(), 0.0);
	}
	differentials[iSize-1] = dErr_dXlast;  // last one
	
	// now iterate through all layers including the last but excluding the first, and ask each of
	// them to backpropagate error and adjust their weights, and to return the differential
	// dErr_dXnm1 for use as the input value of dErr_dXn for the next iterated layer
	for (size_t i = iSize - 1; i > 0; i--)
		m_Layers[i]->BackPropagate_Layer(differentials[i], differentials[i-1], m_etaLearningRate);
}

void NNLayer::BackPropagate_Layer(vector<double>& dErr_dXn   /* in */, 
								  vector<double>& dErr_dXnm1 /* out */,
								  const double eta)
{
	// Err is output error of the entire neural net
	// Xn is the output vector on the n-th layer
	// Xnm1 is the output vector of the previous layer
	const size_t neuronSize = m_Neurons.size();
	const size_t weightSize = m_Weights.size();
	assert(dErr_dXn.size() == neuronSize);
	assert(m_pPrevLayer);
	assert(dErr_dXnm1.size() == m_pPrevLayer->m_Neurons.size());
	
	vector<double> dErr_dYn(neuronSize);

	vector<double> dErr_dWn(weightSize, 0.0);	// important to initialize to zero
	for (size_t i = 0; i < weightSize; ++i)
		dErr_dWn[i] =0.0;
	
	// 1. calculate dErr_dYn = F'(Yn) * dErr_Xn
	for (size_t i = 0; i < neuronSize; ++i) {
		assert(i < dErr_dYn.size());
		assert(i < dErr_dXn.size());
		dErr_dYn[i] = DSIGMOID(m_Neurons[i].output) * dErr_dXn[i];
	}
	
	// 2. calculate dErr_Wn = Xnm1 * dErr_Yn
	for (size_t neuIdx = 0; neuIdx < neuronSize; neuIdx++) {
		for (auto conn_It = m_Neurons[neuIdx].m_Connections.begin(); conn_It < m_Neurons[neuIdx].m_Connections.end(); conn_It++) {
			assert(conn_It->NeuronIndex < m_pPrevLayer->m_Neurons.size());
			assert(conn_It->WeightIndex < weightSize); // m_Weights.size() == dErr_dWn.size()
			assert(neuIdx < dErr_dYn.size());
			dErr_dWn[conn_It->WeightIndex] += dErr_dYn[neuIdx] * m_pPrevLayer->m_Neurons[conn_It->NeuronIndex].output;
			// * calculate dErr_dXnm1 = Wn * dErr_dYn, the previous layer dErr
			if (conn_It == m_Neurons[neuIdx].m_Connections.end()-1)	// optional: ignore dErr calculation on BIAS, if calculated the value will never be used
				continue;
			dErr_dXnm1[conn_It->NeuronIndex] += dErr_dYn[neuIdx] * m_Weights[conn_It->WeightIndex];
		}
	}
	
	// 3. update the weights of this layer neuron using dErr_dW and the learning rate eta
	for (unsigned j = 0; j < weightSize; ++j) {
		m_Weights[j] -= eta * dErr_dWn[j];
	}
}
