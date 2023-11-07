''' ----- Backpropagation code ------'''
import utils

dataset = utils.annDataset("dadosmamografia.csv")

ann = utils.artifialNetwork(5, 50, 1)

utils.trainAnnBackpropagate(100, 5, ann, dataset, 0.001, 1)

# ann.getAnnWeights()

utils.simulateAnn(ann, dataset, generatedImageMaxLenght=15)