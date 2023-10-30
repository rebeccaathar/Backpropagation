''' ----- Backpropagation code ------'''
import utils

dataset = utils.annDataset("dadosmamografia.csv")

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

print(ann.getAnnWeights())