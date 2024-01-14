import numpy as np

from development import Development
from sklearn.utils import shuffle

DEBUG: bool = False


class LogisticRegression:
    def __init__(self, h=0.0001, l=0.001, epochs=800):
        self.h = h
        self.l = l
        self.epochs = epochs

        self.weights = None  # weights vector

    def _sigmoid(self, x):
        if x < 0:
            return np.exp(x) / (1 + np.exp(x))
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X, y):
        n_data, n_features = X.shape
        self.weights = np.random.randn(n_features) # start with random weights

        for epoch in range(self.epochs):
            if DEBUG:
                total_loss = 0
                print(epoch)

            # Shuffle training examples
            X, y = shuffle(X, y, random_state=epoch)
            for i in range(n_data):
                # Compute prediction and gradient loss
                # for current training example
                prediction = self._sigmoid(np.dot(self.weights, X[i]))
                error = y[i] - prediction
                gradient_loss = error * X[i]

                # Calculate gradient for L2 regularization
                gradient_regularization = 2 * self.l * self.weights

                # Calculate weight update
                # based on gradient loss
                # and gradient regularization
                weight_update = self.h * (gradient_loss - gradient_regularization)
                self.weights += weight_update

                if DEBUG:
                    # Add loss for monitoring
                    total_loss += error**2

                if np.isnan(np.sum(self.weights)):
                    print("NaN values detected during training.")
                    print(f"epoch {epoch} i {i}")
                    return

            if DEBUG:
                # Print average loss for monitoring
                average_loss = total_loss / n_data
                print(f"Epoch {epoch}, Average Loss: {average_loss}")

    def predict(self, x):
        predicted_classes = []
        for i in range(len(x)):
            prediction = self._sigmoid(np.dot(self.weights, x[i]))
            predicted_classes.append(1 if prediction > 0.5 else 0)

        return np.array(predicted_classes)


def main():
    development = Development()

    development.calculate(LogisticRegression(h=0.0001, l=0.001, epochs=800))


if __name__ == "__main__":
    main()
