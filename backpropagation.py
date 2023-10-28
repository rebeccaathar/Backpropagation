''' ----- Backpropagation code ------'''
import utils

dataset = "dadosmamografia.csv"
data = utils.load_csv(dataset)
utils.min_max_normalize(data)

neuron = utils.neuron([0.5, 0, 4, 9], +3, utils.neuronActivationFunctions.sigmoide, 1)
print(neuron.net([1, 1, 1, 1]), neuron.activation([1, 1, 1, 1]))

ann = utils.artifialNetwork(1, 2, 1)