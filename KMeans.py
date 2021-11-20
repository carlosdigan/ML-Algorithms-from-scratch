import numpy as np

class KMeans:
    def __init__(self, X, num_centroids):
        self.X = X
        self.num_centroids = num_centroids
        #Select random X points as centroids' initial location
        self.centroids = np.array(X[np.random.randint(len(X), size=num_centroids)], dtype="float") 
        self.distances = np.zeros((len(X), len(self.centroids)))
        
    def findClosestCentroids(self, X, centroids):
        for sample_index, sample in enumerate(X):
            for centroid_index, centroid in enumerate(centroids):
                self.distances[sample_index, centroid_index] = np.linalg.norm(sample - centroid) ** 2
        return self.distances.argmin(axis=1)
    

    def computeCentroids(self, X, indexes):
        for i in range(self.num_centroids):
            self.centroids[i, :] = X[indexes == i].mean(axis=0)
     
        
    def fit(self, iterations=100):
        iteration = 1
        while iteration <= iterations:
            indexes = self.findClosestCentroids(self.X, self.centroids)
            self.computeCentroids(self.X, indexes)
            iteration += 1
      
