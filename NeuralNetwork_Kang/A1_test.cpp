#include "NeuralNetwork.h"
#include "TrainNeuralNetwork.h"
#include "Predict.h"

#include <vector>
using namespace std;

vector<size_t> transNN(size_t* array, size_t num) {
	vector<size_t> ans;
	for (int i = 0; i < num; i++)
		ans.push_back(array[i]);

	return ans;
}

int main()
{
	size_t neuralLayer[] = {3, 5, 8, 7, 9, 2};
	NeuralNetwork NN(0.05f);
	vector<size_t> nnNodes = transNN(neuralLayer, sizeof(neuralLayer)/sizeof(neuralLayer[0]));
	NN.create(nnNodes);

	vector<Data> trainSet;

	TrainNeuralNetwork trainNN(0.1f, 1000u);
	trainNN.Train(NN, &trainSet);

	PredictNN pdtNN;
	Data actual;
	pdtNN.myPredict(NN, actual);
	return 0;
}