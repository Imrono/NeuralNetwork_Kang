A1_test.exe : NeuralNetwork.o TrainNeuralNetwork.o A1_test.o
	gcc NeuralNetwork.o TrainNeuralNetwork.o A1_test.o -std=c++11 -lstdc++ -o A1_test.exe

A1_test.o : A1_test.cpp
	gcc -c A1_test.cpp -std=c++11 -lstdc++ -o A1_test.o

NeuralNetwork.o : NeuralNetwork.h NeuralNetwork.cpp
	gcc -c NeuralNetwork.cpp -std=c++11 -lstdc++ -o NeuralNetwork.o
	
TrainNeuralNetwork.o : NeuralNetwork.h NeuralNetwork.cpp
	gcc -c TrainNeuralNetwork.cpp -std=c++11 -lstdc++ -o TrainNeuralNetwork.o

clean:
	rm NeuralNetwork.o TrainNeuralNetwork.o A1_test.o A1_test.exe