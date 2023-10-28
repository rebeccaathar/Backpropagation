''' ----- Backpropagation code ------'''
import utils

dataset = "dadosmamografia.csv"
data = utils.load_csv(dataset)
utils.min_max_normalize(data)

neuron = utils.neuron([0.5, 0, 4, 9], +3, utils.neuronActivationFunctions.sigmoide, 1)
print(neuron.net([1, 1, 1, 1]), neuron.activation([1, 1, 1, 1]))

ann = utils.artifialNetwork(3, {
    'numberOfHiddenNeurons': 2,
    'weights': [[1,2,3],
                [4,5,6]],
    'activationFunctions': [utils.neuronActivationFunctions.linear, utils.neuronActivationFunctions.hiperbolica],
    'activationFunctionAngularFactor': [1, 2],
    'bias': 10
    }, {
    'numberOfOutputNeurons': 2,
    'weights': [[1,2],
                [4,5]],
    'activationFunctionAngularFactor': [1, 2],
    'bias': 10
    })

print(ann.forward_propagate([1,2,3]))