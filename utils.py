''' ----- Backpropagation code ------'''
import numpy as np
import pandas as pd
from enum import Enum
import random

# Calculate the accuracy of our model
def accuracy_metric(actual, predicted):
 correct = 0
 for i in range(len(actual)):
    if actual[i] == predicted[i]:
        correct += 1
 return correct / float(len(actual)) 

# ANN Dataset class
class annDataset():
    testDataset = pd.DataFrame()
    validationdataset = pd.DataFrame()

    def __init__(self, pathToDataset, testSplitFraction=0.8, ValidationSplitfraction=0.2):
        datasetNotNormalized = self.__loadCsv__(pathToDataset)
        NormalizedDataset = self.__minMaxNormalize__(datasetNotNormalized)
        self.testDataset, self.validationdataset = self.__splitDataframeRandomly__(NormalizedDataset, testSplitFraction, ValidationSplitfraction)

    # Load dataset file
    def __loadCsv__(self, pathToDataset):
        data = pd.read_csv(pathToDataset, sep=';')
        return data

    # Min-Max Normalization
    def __minMaxNormalize__(self, data):
        features = data.iloc[:, :-1]
        min_val = np.min(features, axis=0)
        max_val = np.max(features, axis=0)
        normalized_features = (features - min_val) / (max_val - min_val)
        normalized_features['out'] = data.iloc[:,-1]
        return normalized_features

    # Split dataset in test and validation and use randon sort to change test and validation detaset every time is called
    def __splitDataframeRandomly__(self, dataframe, proportionTest, proportionValidation):
        if (proportionValidation + proportionTest > 1.0): raise ValueError('Test Proportion and Validation proportion must be less or equal 1')
        num_rows = len(dataframe)
        testRows = int(proportionTest * num_rows)
        ValidationRows = int(proportionValidation * num_rows)
        indices = list(dataframe.index)
        random.shuffle(indices)
        testDataset = dataframe.iloc[indices[:testRows], :]
        validationDataset = dataframe.iloc[indices[testRows:ValidationRows+testRows], :]
        return testDataset, validationDataset

# Neuron function activation
class neuronActivationFunctions(Enum):
    def linear(x, p=1):
        return p*x
    
    def sigmoide(x, p=1):
        return (1/(1+np.exp(-1*x*p)))

    def hiperbolica(x, p=1):
        return ((1-np.exp(-1*x*p))/(1+np.exp(-1*x*p)))

    def arctan(x, p=1):
        return np.arctan(p*x)

# The neuron class
class neuron:
    bias = 0
    weights = []
    activationFunctionAngularFactor = 1
    def activationFunction(): pass
    
    # Runtime vars that helps during training:
    lastNetMeasured = 0
    lastOutputMeasured = 0
    lastActivationFunctionsDerived = 0

    # Neuron contructor, it receives, 4 arguments, thus, they are described below.
    # weights: indicates the neuron link weights. E.g: [w1, w2, w3, ..., wN]
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
        net = self.bias #bias 
        for i in range(len(self.weights)):
            net += self.weights[i] * inputs[i]
        self.lastNetMeasured = net
        return net
    
    # Activation is a function that calculates the output of the neuron 
    # based on its net and activation function.
    def activation(self, inputs):
        net = self.net(inputs)
        output = self.activationFunction(net, self.activationFunctionAngularFactor)
        self.lastActivationFunctionsDerived = self.getNeuronActivationFunctionsDerived(net)
        self.lastOutputMeasured = output
        return output

    # Get neuron function activation derived
    def getNeuronActivationFunctionsDerived(self, x):
        p = self.activationFunctionAngularFactor
        match self.activationFunction:
            case neuronActivationFunctions.linear:
                return p
            case neuronActivationFunctions.sigmoide:
                return p*neuronActivationFunctions.sigmoide(x)*(1-neuronActivationFunctions.sigmoide(x))
            case neuronActivationFunctions.hiperbolica:
                return p*(1-(neuronActivationFunctions.hiperbolica(x)^2))
            case neuronActivationFunctions.arctan:
                return (p*(1/(1+x^2)))

# The ANN class 
class artifialNetwork:

    inputLayerNeurons = []
    hiddenLayerNeurons = []
    outputLayerNeurons = []

    # Ariticial neural network constructor. To create this ANN we have some possibilitys
    # that are defined by the parameters.
    # numberOfInputNeurons: is the amount of inputs of the ANN.
    # hiddenNeurons: 
    #   - in first case this parameter can be just a integer that indicates how many neurons
    #     will exist in the hidden layer.
    #   - in second case this can be a dictionary with custom settings to hidden layer, E.g:
    #       {
    #           "numberOfHiddenNeurons" = x, <- just a integer that indicates how many neurons
    #                                          will exist in the hidden layer (REQUIRED)
    #
    #           "weights" = [[w11, w22, ..., w1N],  <- each line represents a unique neuron and it's weights. (OPTIONAL)
    #                        [w21, w22, ..., w2N],
    #                          .     .         .
    #                          .     .         .
    #                          .     .         .
    #                        [wM1, wM2, ..., wMN]],
    #
    #           "activationFunctions" = [activationFunction1, ...,activationFunctionM], <- each column can be intepreted as 
    #                                                                                      a activation function that will be
    #                                                                                      insert in hidden layer neurons. If,
    #                                                                                      just a UNIQUE function is defined, all
    #                                                                                      neurons will have the same activation Function. (OPTIONAL)
    #
    #           "activationFunctionAngularFactor" = [p1, ..., pM], <- each column can be intepreted as 
    #                                                                 a activation function angular factor that will be
    #                                                                 insert in hidden layer neurons. If,
    #                                                                 just a UNIQUE angular factor is defined, all
    #                                                                 functions will have the same angular factor. (OPTIONAL)
    #
    #           "bias" = b <- this is the common bias for all hidden layers neurons (OPTIONAL)
    #       }
    # outputNeurons:
    #   - in first case this parameter can be just a integer that indicates how many neurons
    #     will exist in the output layer.
    #   - in second case this can be a dictionary with all custom settings to output layer, E.g:
    #       {
    #           "numberOfOutputNeurons" = x, <- just a integer that indicates how many neurons
    #                                          will exist in the output layer (REQUIRED)
    #
    #           "weights" = [[w11, w22, ..., w1N],  <- each line represents a unique neuron an it's weights. (OPTIONAL)
    #                        [w21, w22, ..., w2N],
    #                          .     .         .
    #                          .     .         .
    #                          .     .         .
    #                        [wM1, wM2, ..., wMN]],
    #
    #           "activationFunctionAngularFactor" = [p1, ..., pM], <- each column can be intepreted as 
    #                                                                 a activation function angular factor that will be
    #                                                                 insert in output layer neurons. If,
    #                                                                 just a UNIQUE angular factor is defined, all
    #                                                                 functions will have the same angular factor. (OPTIONAL)
    #
    #           "bias" = b <- this is the common bias for all output layer neurons (OPTIONAL)
    #       }
    def __init__(self, numberOfInputNeurons, hiddenNeurons, outputNeurons):
        # Input layer config
        if(numberOfInputNeurons <= 0 or type(numberOfInputNeurons) is not int): raise ValueError('Number of inputs neurons invalid. Parameter must be an integer greater or equals 0')
        for i in range(numberOfInputNeurons):
            self.inputLayerNeurons.append(neuron([1], 0, neuronActivationFunctions.linear, 1))
        
        # Hidden layer config
        if((type(hiddenNeurons) is not int) and (type(hiddenNeurons) is not dict)):
            raise ValueError('Hidden neurons Parameter must be an integer or dictionary')
        if (type(hiddenNeurons) is int):
            if(hiddenNeurons <= 0): raise ValueError('Number of hidden neurons invalid. Parameter must be greater or equals 0')
            hiddenNeurons = {"numberOfHiddenNeurons":hiddenNeurons}
        hiddenNeuronsCompleteSettings = self.__completeHiddenNeuronsSettings__(hiddenNeurons)
        self.__createHiddenLayerFromSetings__(hiddenNeuronsCompleteSettings)

        #output layer config
        if((type(outputNeurons) is not int) and (type(outputNeurons) is not dict)):
            raise ValueError('Output neurons Parameter must be an integer or dictionary')
        if (type(outputNeurons) is int):
            if(outputNeurons <= 0): raise ValueError('Number of output neurons invalid. Parameter must be greater or equals 0')
            outputNeurons = {"numberOfOutputNeurons":outputNeurons}
        outputNeuronsCompleteSettings = self.__completeOutputNeuronsSettings__(outputNeurons)
        self.__createOutputLayerFromSetings__(outputNeuronsCompleteSettings)

    # This method receive an incomplete dictionary with custom hidden layer settings
    # and complete it with default values for optionals not provided.
    def __completeHiddenNeuronsSettings__(self, incompleteDictSettings):
        defaultSettings = {
            "numberOfHiddenNeurons": 0,
            "weights": [],
            "activationFunctions": [],
            "activationFunctionAngularFactor": [], 
            "bias": 0
        }

        # Overwrite default number of hidden neurouns
        if ('numberOfHiddenNeurons' not in incompleteDictSettings):
            raise ValueError('Missing required option numberOfHiddenNeurons')
        else:
            if(incompleteDictSettings['numberOfHiddenNeurons'] <= 0): raise ValueError('Number of hidden neurons invalid. Parameter must be greater or equals 0')
            defaultSettings['numberOfHiddenNeurons'] = incompleteDictSettings['numberOfHiddenNeurons']

        # Overwrite default weights
        if ('weights' not in incompleteDictSettings):
            for i in range(defaultSettings['numberOfHiddenNeurons']):
                defaultSettings['weights'].append(np.random.rand(len(self.inputLayerNeurons)))
        else:
            if(len(incompleteDictSettings['weights'][0]) is not len(self.inputLayerNeurons)): 
                raise ValueError('hidden layer weights column dimension does not match input layer')
            if(len(incompleteDictSettings['weights']) is not defaultSettings['numberOfHiddenNeurons']): 
                raise ValueError('hidden layer weights rows dimension does not match provisioned numberOfHiddenNeurons')
            defaultSettings['weights'] = incompleteDictSettings['weights']
        
        # Overwrite default activation functions
        if ('activationFunctions' not in incompleteDictSettings):
            for i in range(defaultSettings['numberOfHiddenNeurons']):
                defaultSettings['activationFunctions'].append(neuronActivationFunctions.sigmoide)
        else:
            if (type(incompleteDictSettings['activationFunctions']) is not list):
                for i in range(defaultSettings['numberOfHiddenNeurons']):
                    defaultSettings['activationFunctions'].append(incompleteDictSettings['activationFunctions'])
            else:
                if(len(incompleteDictSettings['activationFunctions']) is not defaultSettings['numberOfHiddenNeurons']): 
                    raise ValueError('activationFunctions dimension does not match provisioned numberOfHiddenNeurons')
                defaultSettings['activationFunctions'] = incompleteDictSettings['activationFunctions']

        # Overwrite default activation functions angular factor
        if ('activationFunctionAngularFactor' not in incompleteDictSettings):
            for i in range(defaultSettings['numberOfHiddenNeurons']):
                defaultSettings['activationFunctionAngularFactor'].append(1)
        else:
            if (type(incompleteDictSettings['activationFunctionAngularFactor']) is not list):
                for i in range(defaultSettings['numberOfHiddenNeurons']):
                    defaultSettings['activationFunctionAngularFactor'].append(incompleteDictSettings['activationFunctionAngularFactor'])
            else:
                if(len(incompleteDictSettings['activationFunctionAngularFactor']) is not len(defaultSettings['activationFunctions'])): 
                    raise ValueError('activationFunctionAngularFactor dimension does not match provisioned activationFunctions')
                defaultSettings['activationFunctionAngularFactor'] = incompleteDictSettings['activationFunctionAngularFactor']

        # Overwrite default bias
        if ('bias' in incompleteDictSettings):
            defaultSettings['bias'] = incompleteDictSettings['bias']
        
        return defaultSettings

    # This method create the hidden layer from custom settings provided
    def __createHiddenLayerFromSetings__(self, hiddenNeuronsSettings):
        for i in range(hiddenNeuronsSettings['numberOfHiddenNeurons']):
            self.hiddenLayerNeurons.append(neuron(
                hiddenNeuronsSettings['weights'][i],
                hiddenNeuronsSettings['bias'],
                hiddenNeuronsSettings['activationFunctions'][i],
                hiddenNeuronsSettings['activationFunctionAngularFactor'][i]
            ))
    
    # This method receive an incomplete dictionary with custom output layer settings
    # and complete it with default values for optionals not provided.
    def __completeOutputNeuronsSettings__(self, incompleteDictSettings):
        defaultSettings = {
            "numberOfOutputNeurons": 0,
            "weights": [],
            "activationFunctions": [],
            "activationFunctionAngularFactor": [], 
            "bias": 0
        }

        # Overwrite default number of output neurouns
        if ('numberOfOutputNeurons' not in incompleteDictSettings):
            raise ValueError('Missing required option numberOfOutputNeurons')
        else:
            if(incompleteDictSettings['numberOfOutputNeurons'] <= 0): raise ValueError('Number of output neurons invalid. Parameter must be greater or equals 0')
            defaultSettings['numberOfOutputNeurons'] = incompleteDictSettings['numberOfOutputNeurons']

        # Overwrite default weights
        if ('weights' not in incompleteDictSettings):
            for i in range(defaultSettings['numberOfOutputNeurons']):
                defaultSettings['weights'].append(np.ones(len(self.hiddenLayerNeurons)))
        else:
            if(len(incompleteDictSettings['weights'][0]) is not len(self.hiddenLayerNeurons)): 
                raise ValueError('output layer weights column dimension does not match hidden layer')
            if(len(incompleteDictSettings['weights']) is not defaultSettings['numberOfOutputNeurons']): 
                raise ValueError('output layer weights rows dimension does not match provisioned numberOfOutputNeurons')
            defaultSettings['weights'] = incompleteDictSettings['weights']
        
        # Overwrite default activation functions
        if ('activationFunctions' not in incompleteDictSettings):
            for i in range(defaultSettings['numberOfOutputNeurons']):
                defaultSettings['activationFunctions'].append(neuronActivationFunctions.linear)

        # Overwrite default activation functions angular factor
        if ('activationFunctionAngularFactor' not in incompleteDictSettings):
            for i in range(defaultSettings['numberOfOutputNeurons']):
                defaultSettings['activationFunctionAngularFactor'].append(1)
        else:
            if (type(incompleteDictSettings['activationFunctionAngularFactor']) is not list):
                for i in range(defaultSettings['numberOfOutputNeurons']):
                    defaultSettings['activationFunctionAngularFactor'].append(incompleteDictSettings['activationFunctionAngularFactor'])
            else:
                if(len(incompleteDictSettings['activationFunctionAngularFactor']) is not len(defaultSettings['activationFunctions'])): 
                    raise ValueError('activationFunctionAngularFactor dimension does not match provisioned activationFunctions')
                defaultSettings['activationFunctionAngularFactor'] = incompleteDictSettings['activationFunctionAngularFactor']

        # Overwrite default bias
        if ('bias' in incompleteDictSettings):
            defaultSettings['bias'] = incompleteDictSettings['bias']
        
        return defaultSettings

    # This method create the output layer from custom settings provided
    def __createOutputLayerFromSetings__(self, outputNeuronsSettings):
        for i in range(outputNeuronsSettings['numberOfOutputNeurons']):
            self.outputLayerNeurons.append(neuron(
                outputNeuronsSettings['weights'][i],
                outputNeuronsSettings['bias'],
                outputNeuronsSettings['activationFunctions'][i],
                outputNeuronsSettings['activationFunctionAngularFactor'][i]
            ))

    # Forward propagate information from input to a network output.
    def forwardPropagate(self, data):
        if(len(data) is not len(self.inputLayerNeurons)):
            raise ValueError('provisioned data must have the same dimension from ANN inputs')
        
        # input layer initial propagation
        inputs = []
        for i in range(len(self.inputLayerNeurons)):
            inputs.append(self.inputLayerNeurons[i].activation([data[i]]))
        # other layers propagation
        layers = [self.hiddenLayerNeurons, self.outputLayerNeurons]
        for layer in layers:
            new_inputs = []
            for neuron in layer:
                new_inputs.append(neuron.activation(inputs))
            inputs = new_inputs
        return inputs

    # Get all layers neurons weights
    def getAnnWeights(self):
        layers = [self.inputLayerNeurons, self.hiddenLayerNeurons, self.outputLayerNeurons]
        i = 0
        for layer in layers:
            i += 1
            print(i,'# Layer', sep='')
            j = 0
            for neuron in layer:
                j += 1
                print(neuron.weights, ' <- neuron ', j, '#', sep='')
            print('--------')

# The train method
def trainAnnBackpropagate(artifialNetwork, annDataset, learningFactor, numberOfOutputs=1):
    outputError = []
    for uniqueData in annDataset.testDataset.iloc:
        uniqueData = uniqueData.to_numpy()
        uniqueDataOutputs = uniqueData[-1 * numberOfOutputs :]
        uniqueDataInputs = uniqueData[: -1 * numberOfOutputs]

        # Output Layer weights train
        output = artifialNetwork.forwardPropagate(uniqueDataInputs)
        outputGradient = []
        error = []

        for i in range(len(output)):
            currentError = uniqueDataOutputs[i] - output[i]
            error.append(currentError)
            gradient = (currentError)*artifialNetwork.outputLayerNeurons[i].lastActivationFunctionsDerived
            outputGradient.append(gradient)
        
        # Store old output neurons due to hidden layer trianing
        beforeUpdateOutputNeurons = artifialNetwork.outputLayerNeurons
        for outputNeuron, gradient, i in zip(artifialNetwork.outputLayerNeurons, outputGradient, range(len(outputGradient))):
            updatedWeight = []
            for weight, hiddenNeuron in zip(outputNeuron.weights, artifialNetwork.hiddenLayerNeurons):
                dW = gradient*learningFactor*hiddenNeuron.lastOutputMeasured
                weight += dW
                updatedWeight.append(weight)
            artifialNetwork.outputLayerNeurons[i].weights = updatedWeight
        
        # Total output error
        outputError.append(error)

        # Hidden layer weights training
        localGradient = []

        for i in range(len(artifialNetwork.hiddenLayerNeurons)):
            outputBackpropagatedError = 0
            for outputNeuron, outputRespectiveGradient in zip(beforeUpdateOutputNeurons, outputGradient):
                outputBackpropagatedError += outputNeuron.weights[i]*outputRespectiveGradient
            gradient = outputBackpropagatedError*artifialNetwork.hiddenLayerNeurons[i].lastActivationFunctionsDerived
            localGradient.append(gradient)
        for hiddenNeuron, gradient, i in zip(artifialNetwork.hiddenLayerNeurons, localGradient, range(len(localGradient))):
            updatedWeight = []
            for weight, inputNeuron in zip(hiddenNeuron.weights, artifialNetwork.inputLayerNeurons):
                dW = gradient*learningFactor*inputNeuron.lastOutputMeasured
                weight += dW
                updatedWeight.append(weight)
            artifialNetwork.hiddenLayerNeurons[i].weights = updatedWeight