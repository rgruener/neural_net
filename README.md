neural_net
==========

An implementation of a neural network in C++

![Image](http://www.texample.net/media/tikz/examples/PNG/neural-network.png)

## Description ##

The goal of this project was to make a working represention of a neural
network for use in Machine Learning.  Taking a text file representing 
features of a data set for both training and testing, this neural net
performs multivariate classification.  It should be noted that this 
implementation only allows for a single hidden layer though since this limitation
is in the loading of data files as opposed to back propogation it should be
extremely easy to add the possibility of more hidden layers rather easily.

## Dependencies ##

* gcc
* Make

## Compilation ##

As long as the make utility is installed, simply type make in the root directory
to compile project

## Usage ##

To use, follow text prompts to perform either the training of a neural net or
the testing of a neural net.  To train a neural net a text file is needed to
represent the training data set.  The first line of the file should contain the
number of training examples followed by the number of features in the data set and
the number of possible classes.  Each subsequent line should contain a single training
example consisting of the individual features follower by a list of 0's and 1's indicating
which classes (if any) the current training example belongs to.  An example of a training
file can be found in genre.train.

An initialization file must be created to define the topology of the neural net.  This can
be generated using the python script generate.py.  To use it simply change the 
output filename variable as well as the list of layer sizes before running the script.  This
file can also be created manually.

Testing files follow the same format as training files.

Upon training an output file is created representing the weights of the neural net.
This file should be used when testing.  Upon testing the neural net an output file
is created representing the performance of the neural net.  The first N lines of the
output file correspond to the N possible classes and the numbers represent the prediction
statistics (A, B, C, D represent if 1 was expected and 1 was predicted, 0 was expected and 1
was predicted, etc) as well as overall accuracy, precision, recall, and F1.
