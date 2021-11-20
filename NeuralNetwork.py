import numpy as np

class NeuralNetwork:
    def __init__(self, X, y, nodes, learning_rate=0.001):
        self.num_weights = len(nodes) - 1
        self.lr = learning_rate
        self.all_weights = self.randInitWeights(nodes)
        self.all_bias = self.InitBias(nodes)
        self.X = X.T
        self.y = self.hot_encode(y)
        #self.y = np.array([[0, 0]])



    @staticmethod
    def hot_encode(y):
        m = len(y)
        num_labels = len(np.unique(y))
        Y = np.zeros((m, num_labels))
        for i in range(len(y)):
            Y[i, (int(y[i][0]))] = 1
        return Y.T



    def randInitWeights(self, nodes):
        all_weights = []
        for i in range(self.num_weights):
	    #Using He weight initialization
            all_weights.append(np.random.rand(nodes[i + 1], nodes[i]) * np.sqrt(2 / nodes[i]))
        return all_weights

    def InitBias(self, nodes):
        all_bias = []
        for i in range(self.num_weights):
            all_bias.append(np.zeros((nodes[i + 1], 1)))
        return all_bias

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
        zSig = self.sigmoid(z)
        return zSig * (1 - zSig)

    def forwardPropagation(self, return_layer_outputs=False):
        curr_layer = self.X
        Z = [curr_layer]
        A = [curr_layer]

        for weights, bias in zip(self.all_weights, self.all_bias):
            curr_layer = np.dot(weights, curr_layer) + bias
            Z.append(curr_layer)
            curr_layer = self.sigmoid(curr_layer)
            A.append(curr_layer)

        output_layer = A.pop()
        if return_layer_outputs:
            return Z, A, output_layer
        return output_layer

    def cost(self):
        output_layer = self.forwardPropagation()
        m = self.X.shape[1]
        error = -self.y * np.log(output_layer) - (1 - self.y) * np.log(1 - output_layer)
        cost = (1 / m) * np.sum(error)
        return cost

    def backPropagation(self, iterations=1000):
        m = self.X.shape[1]
        iteration = 1
        while iteration <= iterations:
            Z, A, output_layer = self.forwardPropagation(True)
            dz = output_layer - self.y
            Z.pop() #Get rid of unused last layer output

            for i in range(1, self.num_weights + 1):
                a = A.pop()

                dw = 1 / m * np.dot(dz, a.T)
                db = 1 / m * np.sum(dz, axis=1, keepdims=True)
                dz = np.dot(self.all_weights[-i].T, dz) * self.sigmoidPrime(Z.pop())
                self.all_weights[-i] = self.all_weights[-i] - self.lr * dw
                self.all_bias[-i] = self.all_bias[-i] - self.lr * db



            iteration += 1


    def predict(self):
        predictions = self.forwardPropagation()
        predictions = predictions.argmax(axis=0)
        return predictions.reshape(-1, 1)
