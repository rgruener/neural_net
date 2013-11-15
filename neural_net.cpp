// Robert Gruener
// neural_net.cpp

#include <cmath>
#include <iostream>
#include <iomanip>
#include "neural_net.h"

neuralNet::neuralNet(std::ifstream &initFile) {
    this->numLayers = 3;
    this->layerSizes.resize(this->numLayers);
    this->layers.resize(this->numLayers);
    for (int i=0; i<numLayers; i++) {
        initFile >> this->layerSizes[i];
        this->layerSizes[i]++; // increment to incorporate bias input
        this->layers[i].resize(this->layerSizes[i]);
    }

    // Set bias inputs
    for (int i=0; i<this->numLayers; i++) {
        this->layers[i][0].activation = -1;
    }

    // Set weights
    for (int curLayer=0; curLayer<this->numLayers-1; curLayer++) {
        for (int j=1; j<this->layerSizes[curLayer+1]; j++) {
            for (int k=0; k<this->layerSizes[curLayer]; k++) {
                double weight;
                initFile >> weight;
                neuralConnection incomingConnection, outgoingConnection;
                outgoingConnection.weight = weight;
                outgoingConnection.connectedNeuron = &this->layers[curLayer+1][j];
                this->layers[curLayer][k].outgoingConnections.push_back(outgoingConnection);

                incomingConnection.weight = weight;
                incomingConnection.connectedNeuron = &this->layers[curLayer][k];
                this->layers[curLayer+1][j].incomingConnections.push_back(incomingConnection);
            }
        }
    }
}

int neuralNet::train(std::ifstream &trainingDataFile, double learningRate, int numEpochs) {
    int num_examples, num_inputs, num_outputs;
    std::vector<trainingExample> examples;

    // Read in examples to memory
    trainingDataFile >> num_examples >> num_inputs >> num_outputs;
    examples.resize(num_examples);
    for (int i=0; i<num_examples; i++) {
        examples[i].inputs.resize(num_inputs);
        examples[i].outputs.resize(num_outputs);
        for (int j=0; j<num_inputs; j++) {
            trainingDataFile >> examples[i].inputs[j];
        }
        for (int j=0; j<num_outputs; j++) {
            trainingDataFile >> examples[i].outputs[j];
        }
    }


    // Train neural network
    int outputLayerIndex = this->numLayers - 1;
    for (int epoch=0; epoch<numEpochs; epoch++) {
        for (int cur_example=0; cur_example<num_examples; cur_example++) {

            // Copy input vector of single training example to the input nodes of the network
            for (int node_i=0; node_i<num_inputs; node_i++) {
                this->layers[0][node_i+1].activation = examples[cur_example].inputs[node_i]; // Shifted by 1 due to bias input
            }

            // Propogate the inputs forward to compute the outputs
            for (int layer_l=1; layer_l<this->numLayers; layer_l++) {
                for (int node_j=1; node_j<this->layerSizes[layer_l]; node_j++) {
                    this->layers[layer_l][node_j].inputValue = 0;
                    std::vector<neuralConnection>::iterator it;
                    // Looping through all connections in previous layer to node j
                    for (it=this->layers[layer_l][node_j].incomingConnections.begin(); it!=this->layers[layer_l][node_j].incomingConnections.end(); it++) {
                        this->layers[layer_l][node_j].inputValue += it->weight * it->connectedNeuron->activation;
                    }
                    // Computing the activation of the jth node at layer l
                    this->layers[layer_l][node_j].activation = this->activationFunction(this->layers[layer_l][node_j].inputValue);
                }
            }

            // Propogate errors backward from output layer to input layer
            for (int node_j=1; node_j<this->layerSizes[outputLayerIndex]; node_j++) { // for each node j in output layer
                this->layers[outputLayerIndex][node_j].error = this->activationFunctionPrime(this->layers[outputLayerIndex][node_j].inputValue) *
                                                (examples[cur_example].outputs[node_j-1] - this->layers[outputLayerIndex][node_j].activation);
            }
            for (int layer_l=outputLayerIndex-1; layer_l>0; layer_l--) {
                for (int node_i=1; node_i<this->layerSizes[layer_l]; node_i++) {
                    double sum = 0;
                    std::vector<neuralConnection>::iterator it;
                    // Looping through all connections to layer l + 1
                    for (it=this->layers[layer_l][node_i].outgoingConnections.begin(); it!=this->layers[layer_l][node_i].outgoingConnections.end(); it++) {
                        sum += it->weight * it->connectedNeuron->error;
                    }
                    this->layers[layer_l][node_i].error = this->activationFunctionPrime(this->layers[layer_l][node_i].inputValue) * sum;
                }
            }
            for (int layer_l=1; layer_l<this->numLayers; layer_l++) {
                for (int node_j=1; node_j<this->layerSizes[layer_l]; node_j++) {
                    std::vector<neuralConnection>::iterator it;
                    // Looping through all connections in previous layer to node j
                    for (it=this->layers[layer_l][node_j].incomingConnections.begin(); it!=this->layers[layer_l][node_j].incomingConnections.end(); it++) {
                        // Update weights in both directions
                        it->weight = it->weight + learningRate * it->connectedNeuron->activation * this->layers[layer_l][node_j].error;
                        it->connectedNeuron->outgoingConnections[node_j-1].weight = it->weight;
                    }
                }
            }
        }
    }

    return 0;
}

int neuralNet::test(std::ifstream &testingDataFile, std::ofstream &outputFile){
    int num_examples, num_inputs, num_outputs;
    std::vector<trainingExample> examples;
    std::vector<std::vector<double> > results;

    // Read in examples to memory
    testingDataFile >> num_examples >> num_inputs >> num_outputs;
    examples.resize(num_examples);
    results.resize(num_outputs);
    for (int i=0; i<num_examples; i++) {
        examples[i].inputs.resize(num_inputs);
        examples[i].outputs.resize(num_outputs);
        for (int j=0; j<num_inputs; j++) {
            testingDataFile >> examples[i].inputs[j];
        }
        for (int j=0; j<num_outputs; j++) {
            testingDataFile >> examples[i].outputs[j];
            if (i == 0) {
                results[j].resize(4);
                for (int k=0; k<4; k++) {
                    results[j][k] = 0;
                }
            }
        }
    }

    int outputLayerIndex = this->numLayers - 1;
    for (int cur_example=0; cur_example<num_examples; cur_example++) {
        // Copy input vector of single training example to the input nodes of the network
        for (int node_i=0; node_i<num_inputs; node_i++) {
            this->layers[0][node_i+1].activation = examples[cur_example].inputs[node_i]; // Shifted by 1 due to bias input
        }

        // Propogate the inputs forward to compute the outputs
        for (int layer_l=1; layer_l<this->numLayers; layer_l++) {
            for (int node_j=1; node_j<this->layerSizes[layer_l]; node_j++) {
                this->layers[layer_l][node_j].inputValue = 0;
                std::vector<neuralConnection>::iterator it;
                // Looping through all connections in previous layer to node j
                for (it=this->layers[layer_l][node_j].incomingConnections.begin(); it!=this->layers[layer_l][node_j].incomingConnections.end(); it++) {
                    this->layers[layer_l][node_j].inputValue += it->weight * it->connectedNeuron->activation;
                }
                // Computing the activation of the jth node at layer l
                this->layers[layer_l][node_j].activation = this->activationFunction(this->layers[layer_l][node_j].inputValue);
            }
        }

        // Threshold the outputs
        for (int node_i=1; node_i<this->layerSizes[outputLayerIndex]; node_i++){
            if (this->layers[outputLayerIndex][node_i].activation >= 0.5) {
                if (examples[cur_example].outputs[node_i-1]) {
                    results[node_i-1][0]++;
                } else {
                    results[node_i-1][1]++;
                }
            } else {
                if (examples[cur_example].outputs[node_i-1]) {
                    results[node_i-1][2]++;
                } else {
                    results[node_i-1][3]++;
                }
            }
        }
    }

    // Print results to output file
    outputFile << std::setprecision(3) << std::fixed;
    double global_A = 0, global_B = 0, global_C = 0, global_D = 0;
    double avg_overall, avg_precision, avg_recall, avg_f1;
    double overall_accuracy, precision, recall, f1;
    for (int i=0; i<num_outputs; i++) {
        global_A += results[i][0];
        global_B += results[i][1];
        global_C += results[i][2];
        global_D += results[i][3];
        outputFile << (int)results[i][0] << " " << (int)results[i][1] << " " << (int)results[i][2] << " " << (int)results[i][3] << " ";
        overall_accuracy = (results[i][0] + results[i][3]) / (results[i][0] + results[i][1] + results[i][2] + results[i][3]);
        precision = results[i][0] / (results[i][0] + results[i][1]);
        recall = results[i][0] / (results[i][0] + results[i][2]);
        f1 = (2 * precision * recall) / (precision + recall);
        outputFile << overall_accuracy << " " << precision << " " << recall << " " << f1 << std::endl;
        avg_overall += overall_accuracy;
        avg_precision += precision;
        avg_recall += recall;
    }
    // Micro-averaging
    overall_accuracy = (global_A + global_D) / (global_A + global_B + global_C + global_D);
    precision = global_A / (global_A + global_B);
    recall = global_A / (global_A + global_C);
    f1 = (2 * precision * recall) / (precision + recall);
    outputFile << overall_accuracy << " " << precision << " " << recall << " " << f1 << std::endl;
    // Macro-averaging
    avg_overall /= num_outputs;
    avg_precision /= num_outputs;
    avg_recall /= num_outputs;
    avg_f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall);
    outputFile << avg_overall << " " << avg_precision << " " << avg_recall << " " << avg_f1 << std::endl;

    return 0;
}

// Activation function used is the sigmoid as defined in problem description
double neuralNet::activationFunction(double inputValue) {
    return 1.0 / (1.0 + exp(-inputValue));
}

// Derivative of activation function (which is sigmoid) as defined in problem description
double neuralNet::activationFunctionPrime(double inputValue) {
    return this->activationFunction(inputValue) * (1.0 - this->activationFunction(inputValue));
}

void neuralNet::printNetwork(std::ostream &outputFile){
    // Print Neural Network
    outputFile << std::setprecision(3) << std::fixed;
    int outputLayerIndex = this->numLayers - 1;
    for (int layer=0; layer<this->numLayers; layer++) {
        if (layer != 0){
            outputFile << " ";
        }
        outputFile << this->layerSizes[layer]-1;
    }
    outputFile << std::endl;
    for (int layer_l=1; layer_l<this->numLayers; layer_l++) {
        for (int node_j=1; node_j<this->layerSizes[layer_l]; node_j++) {
            std::vector<neuralConnection>::iterator it;
            // Looping through all connections in previous layer to node j
            for (it=this->layers[layer_l][node_j].incomingConnections.begin(); it!=this->layers[layer_l][node_j].incomingConnections.end(); it++) {
                // Print weight
                if (it != this->layers[layer_l][node_j].incomingConnections.begin()) {
                    outputFile << " ";
                }
                outputFile << it->weight;
            }
            outputFile << std::endl;
        }
    }

}
