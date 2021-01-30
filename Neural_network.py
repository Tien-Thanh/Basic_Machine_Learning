import numpy as np

def sigmoid(x):
    return 1/ (1 + np.exp(-x))

def sigmoid_der(x):
    return x * (1- x)

class NeuralNetwork:
    def __init__(self, layers, alpha = 0.1):
        self.layers = layers
        self.alpha = alpha
        self.W = []
        self.b = []

        for i in range(0, len(layers) - 1):
            w_ = np.random.randn(layers[i], layers[i+1])
            b_ = np.zeros((layers[i+1], 1))
            self.W.append(w_/layers[i])
            self.b.append((b_))
    def __repr__(self):
        return "Neural Network [{}]".format("-".join(str(1) for l in self.layers))

    def fit_partial(self, x, y):
        A = [x]
        # feedforward
        out = A[-1]
        for i in range(0, len(self.layers) - 1):
            out = sigmoid(np.dot(out, self.W[i]) + (self.b[i].T))
            A.append(out)

        #backpropagation
        y = y.reshape(-1, 1)
        dA = [-(y/A[-1] - (1-y)/(1-A[-1]))]
        dW = []
        db = []
        for i in reversed(range(0, len(self.layers)- 1)):
            dw_ = np.dot((A[i]).T, dA[-1] * sigmoid_der(A[i+1]))
            db_ = (np.sum(dA[-1] * sigmoid_der(A[i+1]), 0)).reshape(-1, 1)
            dA_ = np.dot(dA[-1] * sigmoid_der(A[i+1]), self.W[i].T)
            dW.append(dw_)
            db.append(db_)
            dA.append(dA_)

        dW = dW[::-1]
        db = db[::-1]

        for i in range(0, len(self.layers)-1):
            self.W[i] = self.W[i] - self.alpha * dW[i]
            self.b[i] = self.b[i] - self.alpha * db[i]
    def fit(self, X, y, epochs = 20, verbose = 10):
        for epochs in range(0,epochs):
            self.fit_partial(X, y)
            if epochs % verbose == 0:
                loss = self.calculate_loss(X, y)
                print("Epoch {}, loss {}".format(epochs,loss))

    def predict(self, X):
        for i in range(0, len(self.layers)-1):
            X = sigmoid(np.dot(X, self.W[i]) + (self.b[i].T))
        return X

    def calculate_loss(self, X, y):
        y_predict = self.predict(X)
        return -(np.sum(y*np.log(y_predict) + (1-y)*np.log(1-y_predict)))



