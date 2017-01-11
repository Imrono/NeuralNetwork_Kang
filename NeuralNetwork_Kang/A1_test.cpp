#include "NeuralNetwork.h"
#include "TrainNeuralNetwork.h"
#include "Predict.h"

#include <cstdio>
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
	// Create NN
	printf("### CREATE NN BEGINING\n");
	size_t neuralLayer[] = {3, 18, 18, 3};
	size_t nLayer = sizeof(neuralLayer)/sizeof(neuralLayer[0]);
	printf("input neuralLayer: ");
	for (size_t i = 0; i < nLayer; i++) {
		printf("%u ", neuralLayer[i]);
	}	printf("\n");
	NeuralNetwork NN(0.05f);
	vector<size_t> nnNodes = transNN(neuralLayer, nLayer);
	NN.create(nnNodes);
	printf("### CREATE NN FINISHED\n");

	// Train NN
	printf("### TRAIN NN BEGINING\n");
	vector<Data> trainSet;
	Data trainData;
	trainData.x.push_back(1.0f); trainData.x.push_back(0.0f); trainData.x.push_back(0.0f);
	trainData.y.push_back(1.0f); trainData.y.push_back(0.0f); trainData.y.push_back(0.0f);
	trainSet.push_back(trainData);
	trainData.x.clear(); trainData.y.clear();
	trainData.x.push_back(0.0f); trainData.x.push_back(1.0f); trainData.x.push_back(0.0f);
	trainData.y.push_back(0.0f); trainData.y.push_back(1.0f); trainData.y.push_back(0.0f);
	trainSet.push_back(trainData);
	trainData.x.clear(); trainData.y.clear();
	trainData.x.push_back(0.0f); trainData.x.push_back(0.0f); trainData.x.push_back(1.0f);
	trainData.y.push_back(0.0f); trainData.y.push_back(0.0f); trainData.y.push_back(1.0f);
	trainSet.push_back(trainData);
	trainData.x.clear(); trainData.y.clear();
	trainData.x.push_back(1.0f); trainData.x.push_back(1.0f); trainData.x.push_back(0.0f);
	trainData.y.push_back(1.0f); trainData.y.push_back(1.0f); trainData.y.push_back(0.0f);
	trainSet.push_back(trainData);
	trainData.x.clear(); trainData.y.clear();
	trainData.x.push_back(0.0f); trainData.x.push_back(1.0f); trainData.x.push_back(1.0f);
	trainData.y.push_back(0.0f); trainData.y.push_back(1.0f); trainData.y.push_back(1.0f);
	trainSet.push_back(trainData);
	trainData.x.clear(); trainData.y.clear();
	trainData.x.push_back(1.0f); trainData.x.push_back(0.0f); trainData.x.push_back(1.0f);
	trainData.y.push_back(1.0f); trainData.y.push_back(0.0f); trainData.y.push_back(1.0f);
	trainSet.push_back(trainData);
	trainData.x.clear(); trainData.y.clear();

	TrainNeuralNetwork trainNN(0.00001f, 5000u);
	trainNN.setSamples(trainSet);
	printf("### TRAIN NN FISISHED\n");
	trainNN.Train(NN);
	printf("### TRAIN NN FISISHED\n");

	// NN Predict
	printf("### NN BEGIN TO DO PREDICTION\n");
	PredictNN pdtNN;
	Data actual;
	actual.x.push_back(1.0f); actual.x.push_back(1.0f); actual.x.push_back(1.0f); 
	actual.y.push_back(0.0f); actual.y.push_back(0.0f); actual.y.push_back(0.0f); 
	pdtNN.myPredict(NN, actual);
	printf("1->x:%.5f, %.5f, %.5f\ty:%.5f, %.5f, %.5f\n", actual.x[0], actual.x[1], actual.x[2], actual.y[0], actual.y[1], actual.y[2]);
	actual.x.clear(); actual.y.clear();
	actual.x.push_back(1.0f); actual.x.push_back(0.0f); actual.x.push_back(0.0f); 
	actual.y.push_back(0.0f); actual.y.push_back(0.0f); actual.y.push_back(0.0f); 
	pdtNN.myPredict(NN, actual);
	printf("2->x:%.5f, %.5f, %.5f\ty:%.5f, %.5f, %.5f\n", actual.x[0], actual.x[1], actual.x[2], actual.y[0], actual.y[1], actual.y[2]);
	actual.x.clear(); actual.y.clear();
	actual.x.push_back(0.0f); actual.x.push_back(1.0f); actual.x.push_back(0.0f); 
	actual.y.push_back(0.0f); actual.y.push_back(0.0f); actual.y.push_back(0.0f); 
	pdtNN.myPredict(NN, actual);
	printf("3->x:%.5f, %.5f, %.5f\ty:%.5f, %.5f, %.5f\n", actual.x[0], actual.x[1], actual.x[2], actual.y[0], actual.y[1], actual.y[2]);
	printf("### NN FINISH PREDICTING\n");
	
	
	return 0;
}