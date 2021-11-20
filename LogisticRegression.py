import numpy as np

class LogisticRegression:
    def __init__(self, X, y, learning_rate=0.001, normalize=False):
        self.num_columns = None
        self.num_rows = None
        self.normalize_data = normalize
        self.mean_values = 0
        self.std_values = 0
        self.X = self.insertOnes(X) #Add a column of ones to account for intercept 
        self.y = y.reshape(-1, 1) #Reshape so y is a column vector
        self.lr = learning_rate
        self.theta = np.zeros((self.num_columns, 1))
        
    
    def insertOnes(self, X, train_data=True): 
        if X.ndim == 1: #If X is a 1-D array raise error
            raise TypeError("Input is 1-D array when expected an array of higher dimension")
            
        if self.normalize_data:
            self.normalize(X, train_data)
            
        self.num_rows = X.shape[0]
        ones = np.ones((self.num_rows, 1))
        X = np.hstack((ones, X))
        self.num_columns = X.shape[1]
        return X
    
    
    def normalize(self, X, train_data):
        if train_data:
            self.mean_values = X.mean(axis=0)
            self.std_values = X.std(axis=0, ddof=1)    
            
        return (X - self.mean_values) / self.std_values
    
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
        
    def cost(self):
        #Add a tiny epsilon value to account for undefined log(0)
        error = -self.y * np.log(self.hTheta + 1e-3) - (1 - self.y) * np.log((1 - self.hTheta) + 1e-3) 
        j = 1 / self.num_rows * np.sum(error)
        return j
    
    @property #Make methods getters so they're natural to call
    def hTheta(self):
        return self.sigmoid(np.dot(self.X, self.theta))
    
    @property
    def newTheta(self):
        difference = self.hTheta - self.y
        newTheta = np.dot(self.X.T, difference)
        newTheta = 1/self.num_rows * self.lr * newTheta
        return newTheta
    
    def fit(self, iterations=1000):
        iteration = 1
        #Comment out below line and append line if not tracking costs
        costs = []
        while iteration <= iterations:
            self.theta = self.theta - self.newTheta
            costs.append(self.cost())
            iteration += 1
        return costs
    
    def predict(self, X):
        X = self.insertOnes(X, False)
        predictions = self.sigmoid(np.dot(X, self.theta)) + 0.01 #Add 0.01 so it rounds values at 0.50 up to 1 
        return predictions.round()
    
        
        
