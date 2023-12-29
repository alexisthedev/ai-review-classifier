import numpy as np
from sklearn.utils import shuffle

class LogisticRegression:
    def __init__(self, h=0.001, l=0.01, epochs=1000):
        self.h = h
        self.l = l
        self.epochs = epochs

        self.weights = None # weights vector

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, x_train, y_train):
        n_data, n_features = x_train.shape
        self.weights = np.random.randn(n_features) # start with random weights

        for epoch in range(self.epochs):
            # Shuffle training examples
            x_train, y_train = shuffle(x_train, y_train, random_state=epoch)
            for i in range(n_data):
                # Compute prediction and loss for
                # current training example
                prediction = self._sigmoid(np.dot(self.weights, x_train[i]))
                gradient = (y_train[i] - prediction) * x_train[i]
                # regularization = np.sum(self.weights**2)

                self.weights += (self.h * gradient)

    def predict(self, x):
        predicted_classes = []
        for i in range(len(x)):
            prediction = self._sigmoid(np.dot(self.weights, x[i]))
            predicted_classes.append(1 if prediction > 0.5 else 0)

        return np.array(predicted_classes)
