''' ----- Backpropagation code ------'''
import utils

dataset = "dadosmamografia.csv"
data = utils.load_csv(dataset)
utils.min_max_normalize(data)
