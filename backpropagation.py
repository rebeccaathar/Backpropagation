''' ----- Backpropagation code ------'''
import utils

dataset = utils.annDataset("dadosmamografia.csv")

ann = utils.artifialNetwork(5, 100, 1)

utils.trainAnnBackpropagate(ann, dataset, 0.0001, 1)

ann.getAnnWeights()

utils.simulateAnn(ann, dataset, generatedImageMaxLenght=15)