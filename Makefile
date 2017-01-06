NeuralNetwork.o : NeuralNetwork.h NeuralNetwork.cpp
	gcc -c NeuralNetwork.cpp -std=c++11 -lstdc++ -save-temps -o NeuralNetwork.o

clean:
	rm NeuralNetwork.o