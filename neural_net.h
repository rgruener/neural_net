// Robert Gruener
// neural_net.h


#ifndef _NEURAL_NET_H
#define _NEURAL_NET_H

#include <vector>
#include <fstream>

class neuralNet {

    public:

        neuralNet(std::ifstream &initFile);

        int train(std::ifstream &trainingDataFile, double learningRate, int numEpochs);

        int test(std::ifstream &testingDataFile, std::ofstream &outputFile);

        void printNetwork(std::ostream &outputFile);

    private:

        class neuralConnection;

        class neuron {
            public:
                double activation;
                double inputValue;
                double error;
                std::vector<neuralConnection> incomingConnections;
                std::vector<neuralConnection> outgoingConnections;
        };

        class neuralConnection {
            public:
                double weight;
                neuron *connectedNeuron;
        };

        class trainingExample {
            public:
                std::vector<double> inputs;
                std::vector<int> outputs;
        };

        int numLayers;
        std::vector<int> layerSizes;
        std::vector<std::vector<neuron> > layers;

        double activationFunction(double inputValue);

        double activationFunctionPrime(double inputValue);
};

#endif
