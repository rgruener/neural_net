// Robert Gruener
// main.cpp

#include <iostream>
#include <fstream>
#include <string>
#include "neural_net.h"

using namespace std;

int main() {

    string response, weightsFilename, examplesFilename, outFilename;
    ifstream weightsInfile, examplesInfile;
    ofstream outfile;
    cout << "Would you like to train a neural network? (Y/N): ";
    cin >> response;
    if (response == "Y" || response == "y") {
        int epochs;
        double learning_rate;
        cout << "Please enter filename for intialization weights: ";
        cin >> weightsFilename;
        cout << "Please enter filename for training examples: ";
        cin >> examplesFilename;
        cout << "Please enter filename for output: ";
        cin >> outFilename;
        cout << "Please enter learning rate: ";
        cin >> learning_rate;
        cout << "Please enter number of epochs: ";
        cin >> epochs;
        weightsInfile.open(weightsFilename.c_str());
        examplesInfile.open(examplesFilename.c_str());
        outfile.open(outFilename.c_str());
        if (weightsInfile.is_open() && examplesInfile.is_open() && outfile.is_open()) {
            neuralNet *network = new neuralNet(weightsInfile);
            network->train(examplesInfile, learning_rate, epochs);
            network->printNetwork(outfile);
        } else {
            cerr << "Error while opening input or output file" << endl;
        }
        return 0;
    }
    cout << "Would you like to test a previously trained neural network? (Y/N): ";
    cin >> response;
    if (response == "Y" || response == "y") {
        cout << "Please enter filename for trained weights: ";
        cin >> weightsFilename;
        cout << "Please enter filename for testing examples: ";
        cin >> examplesFilename;
        cout << "Please enter filename for output: ";
        cin >> outFilename;
        weightsInfile.open(weightsFilename.c_str());
        examplesInfile.open(examplesFilename.c_str());
        outfile.open(outFilename.c_str());
        if (weightsInfile.is_open() && examplesInfile.is_open() && outfile.is_open()) {
            neuralNet *network = new neuralNet(weightsInfile);
            network->test(examplesInfile, outfile);
        } else {
            cerr << "Error while opening input or output file" << endl;
        }
        return 0;
    }
    cout << "Then this program is not for you." << endl;
    return 0;
}
