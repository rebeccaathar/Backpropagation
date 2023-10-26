''' ----- Backpropagation code ------'''
import numpy as np
import pandas as pd
from enum import Enum

# Load dataset file
def load_csv(filename):
    data = pd.read_csv(filename, sep=';')
    return data

# Min-Max Normalization
def min_max_normalize(data):
    features = data.iloc[:, :-1]
    min_val = np.min(features, axis=0)
    max_val = np.max(features, axis=0)
    normalized_features = (features - min_val) / (max_val - min_val)
    return normalized_features

# Calculate the accuracy of our model
def accuracy_metric(actual, predicted):
 correct = 0
 for i in range(len(actual)):
    if actual[i] == predicted[i]:
        correct += 1
 return correct / float(len(actual)) 
 
# Linear neuron activation
def linear(x):
    return x

# Forward propagate information from input to a network output.
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
    for neuron in layer:
        activation = activate(neuron['weights'], inputs)
        neuron['output'] = linear(activation)
        new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Neuron function activation
class MyEnum(Enum):
    def linear(x, p=1):
        return p*x
    
    def sigmoide(x, p=1):
        return (1/(1+np.exp(-1*x*p)))

    def hiperbolica(x, p=1):
        return ((1-np.exp(-1*x*p))/(1+np.exp(-1*x*p)))

    def arctan(x, p=1):
        return (p*(1/(1+x^2)))

# The neuron class
class neuron:
    bias = 0
    weights = []
    activationFunctionAngularFactor = 1
    def activationFunction(): pass
    
    # Neuron contructor, it receives, 4 arguments, thus, they are described below.
    # weights: indicates the neuron link weights
    # bias: bias of the neuron
    # activationFunction: define the activation function of the neuron
    # activationFunctionAngularFactor: it is the angular factor of the neuron
    def __init__(self, weights, bias, activationFunction, activationFunctionAngularFactor):
        self.weights = weights
        self.bias = bias
        self.activationFunction = activationFunction
        self.activationFunctionAngularFactor = activationFunctionAngularFactor

    # net is a function that calculates the output of the node 
    # based on its inputs and the weights on individual inputs. 
    def net(self, inputs):
        activation = self.bias #bias 
        for i in range(len(self.weights)):
            activation += self.weights[i] * inputs[i]
        return activation
    
    # Activation is a function that calculates the output of the neuron 
    # based on its net and activation function.
    def activation(self, inputs):
        net = self.net(inputs)
        return self.activationFunction(net, self.activationFunctionAngularFactor)
 