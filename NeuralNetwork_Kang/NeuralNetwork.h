#ifndef _NEURALNETWORK_H_
#define _NEURALNETWORK_H_

#include <cmath>
#include <vector>
#include <climits>
#include <cstdlib>
using namespace std;

#define WEIGTH_INIT (0.05*rand()/RAND_MAX) 		// 初始化权重在0~0.05 
//#define WEIGTH_INIT (0.0f)

// 1
#define SIGMOID(x) (1.7159*tanh(0.66666667*x))
#define DSIGMOID(S) (0.66666667/1.7159*(1.7159+(S))*(1.7159-(S)))  // derivative of the sigmoid as a function of the sigmoid's output
// 2
#define RELU(x) ((x) > 0.0f ? (x) : (0.1f*x))
#define DRELU(S) ((S) > 0.0f ? 1.0f : 0.1f)

// forward declarations
class NNLayer;
class NNNeuron;
struct NNConnection;

class NeuralNetwork  
{
public:
	NeuralNetwork(const double& eta) : m_nLayer(0), m_etaLearningRate(eta) {};
	~NeuralNetwork();
	void reset();
	void create(const vector<size_t>& ar_nodes);	// 创建网络

	void forwardCalc_NN(const vector<double>& inputVector, vector<double>& outputVector);
	void BackPropagate_NN(const double* const actualOutput, const double* const desiredOutput, size_t count);
	
	void forget_NN(double minRate);

	void weightToString();
	void neuronLayerToString();
	
	vector<NNLayer*> m_Layers;	
private: 
	size_t m_nLayer; 				// 网络层数 
    vector<size_t> m_nodes; 		// 每层的结点数 
    vector<double> m_actualOutput;	// 每次迭代的输出结果 
    double m_etaLearningRate; 		// 权值学习率
};

class NNLayer
{
public:
	NNLayer(const size_t id) : layerId(id), m_pPrevLayer(nullptr) {};
	~NNLayer() {};
	
	void addNeurals(size_t num, size_t preNumNeurals, bool isInOut = false);
	
	void forwardCalc_Layer();
	void BackPropagate_Layer(vector<double>& dErr_dXn     /* in */, 
							 vector<double>& dErr_dXnm1   /* out */, 
							 const double eta);
		
	void forget_Layer(double minRate);
	static size_t count;

	double activation(double x) {
		//return SIGMOID(x);
		return RELU(x);
	}
	double d_activation(double x) {
		//return DSIGMOID(x);
		return DRELU(x);
	}
	
	vector<NNNeuron> m_Neurons;
	size_t layerId;
	
	vector<double> m_Weights;
	NNLayer* m_pPrevLayer;
		
public:
	void weightToString();
	void neuronToString();
};

struct NNConnection
{
	size_t NeuronIndex;
	size_t WeightIndex;
};

class NNNeuron
{
public:
	NNNeuron() : output(0.0f) {};
	NNNeuron(const double& biasOutput) : output(biasOutput) {};
	~NNNeuron() {};
	
	double z;
	double output;
	vector<NNConnection> m_Connections;
};

#endif // !defined(_NEURALNETWORK_H_)
