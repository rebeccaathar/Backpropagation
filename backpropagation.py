''' ----- Backpropagation code ------'''
import numpy as np
import pandas as pd
 
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


# Activation is a function that calculates the output of the node 
# based on its inputs and the weights on individual inputs. 
def activate(weights, inputs):
    activation = weights[-1] #bias 
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation
 
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
 
dataset = "dadosmamografia.csv"
data = load_csv(dataset)
min_max_normalize(data)
