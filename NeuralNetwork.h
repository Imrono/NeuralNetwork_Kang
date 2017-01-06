#ifndef _NEURALNETWORK_H_
#define _NEURALNETWORK_H_

#include <cmath>
#include <vector>
#include <climits>
using namespace std;

#define SIGMOID(x) (1.7159*tanh(0.66666667*x))
#define DSIGMOID(S) (0.66666667/1.7159*(1.7159+(S))*(1.7159-(S)))  // derivative of the sigmoid as a function of the sigmoid's output

// forward declarations
class NNLayer;
class NNNeuron;
class NNConnection;

class NeuralNetwork  
{
public:
	NeuralNetwork() : nLayer(0), m_etaLearningRate(0.0f), iterNum(0) {};
	~NeuralNetwork();
	void reset();
	void create(const unsigned numLayers, const unsigned* const ar_nodes);	// 创建网络 

	void forwardCalc_NN(const double* const inputVector, const unsigned count, 
						double* outputVector = nullptr, const unsigned oCount = 0);
	void BackPropagate_NN(double* actualOutput, double* desiredOutput, unsigned count);

	vector<NNLayer*> m_Layers;	
private: 
    unsigned nLayer; 			// 网络层数 
    vector<unsigned> nodes; 	// 每层的结点数 
    vector<double> actualOutput;// 每次迭代的输出结果 
    double m_etaLearningRate; 	// 权值学习率 
    unsigned iterNum; 			// 迭代次数
};

class NNLayer
{
public:
	NNLayer() : m_pPrevLayer(nullptr) {};
	~NNLayer() {};
	
	void addNeurals(unsigned num, unsigned preNumNeurals);
	
	void forwardCalc_Layer();
	void BackPropagate_Layer(vector<double>& dErr_dXn     /* in */, 
							 vector<double>& dErr_dXnm1   /* out */, 
							 const double eta);

	vector<NNNeuron> m_Neurons;
	
	vector<double> m_Weights;
	NNLayer* m_pPrevLayer;
};

struct NNConnection
{
	unsigned NeuronIndex;
	unsigned WeightIndex;
};

class NNNeuron
{
public:
	NNNeuron() : output(0.0f) {};
	NNNeuron(const double& biasOutput) : output(biasOutput) {};
	~NNNeuron() {};
	
	double output;
	vector<NNConnection> m_Connections;
};

#endif // !defined(_NEURALNETWORK_H_)
