''' ----- Backpropagation code ------'''
import utils

dataset = utils.annDataset("dadosmamografia.csv")

ann = utils.artifialNetwork(5, 5, 1)

utils.trainAnnBackpropagate(ann, dataset, 0.1, 1)

ann.getAnnWeights