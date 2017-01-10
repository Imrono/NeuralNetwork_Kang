A1_test.exe : NeuralNetwork.o TrainNeuralNetwork.o A1_test.o Predict.o
	gcc NeuralNetwork.o TrainNeuralNetwork.o A1_test.o Predict.o -std=c++11 -lstdc++ -o A1_test.exe

A1_test.o : A1_test.cpp NeuralNetwork.h TrainNeuralNetwork.h
	gcc -c A1_test.cpp -std=c++11 -lstdc++ -o A1_test.o
	
Predict.o : Predict.h Predict.cpp TrainNeuralNetwork.h NeuralNetwork.h
	gcc -c Predict.cpp -std=c++11 -lstdc++ -o Predict.o

NeuralNetwork.o : NeuralNetwork.h NeuralNetwork.cpp
	gcc -c NeuralNetwork.cpp -std=c++11 -lstdc++ -o NeuralNetwork.o
	
TrainNeuralNetwork.o : TrainNeuralNetwork.h TrainNeuralNetwork.cpp NeuralNetwork.h
	gcc -c TrainNeuralNetwork.cpp -std=c++11 -lstdc++ -o TrainNeuralNetwork.o

clean:
	rm -f NeuralNetwork.o TrainNeuralNetwork.o A1_test.o Predict.o A1_test.exe