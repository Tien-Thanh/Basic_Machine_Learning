import pandas as pd
from Neural_network import *

data = pd.read_csv('dataset_logis.csv').values
N, d = data.shape
x = data[:, 0: d-1].reshape(-1, d-1)
y = data[:, 2].reshape(-1, 1)
x_pre = np.array([[2,10]])
Layers = [2, 2, 1]
net = NeuralNetwork(layers = Layers, alpha=0.01)
net.fit(x, y, epochs = 1000, verbose= 10)
y_pre = net.predict(x_pre)
print(y_pre)