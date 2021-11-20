import numpy as np
from scipy.stats import mode


class KNearestNeighbors:
    def __init__(self, x, y, k=3):
        self.X = x
        self.y = y
        self.k = k

    def euclidean_distance(self, sample):
        distance = np.linalg.norm(sample - self.X, axis=1)
        return distance

    def closest_neighbors(self, distance):
        idx = np.argsort(distance)
        return idx[:self.k]  # Return the first K values

    def get_labels(self, sample):
        distance = self.euclidean_distance(sample)
        idx = self.closest_neighbors(distance)
        true_labels = self.y[idx]
        return true_labels


class KNeighborsClassifier(KNearestNeighbors):
    def __init__(self, X, y, k=3):
        super().__init__(X, y, k)

    def predict(self, samples):
        if samples.dim == 1:
            raise TypeError("Input is 1-D array when expected an array of higher dimension")

        predictions = np.zeros((len(samples), 1))  # Column vector of predictions
        for sample in samples:
            distance = self.euclidean_distance(sample)
            idx = self.closest_neighbors(distance)


class KNeighborsRegressor(KNearestNeighbors):
    def __init__(self, X, y, k=3):
        super().__init__(X, y, k)

    def predict(self, samples):
        if samples.dim == 1:
            raise TypeError("Input is 1-D array when expected an array of higher dimension")

        predictions = np.zeros((len(samples), 1))  # Column vector of predictions

        for index, sample in enumerate(samples):
            true_labels = self.get_labels(sample)
            prediction = np.mean(true_labels)
            predictions[index] = prediction

        return predictions

