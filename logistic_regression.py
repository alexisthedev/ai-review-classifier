import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocess import Preprocess
from sklearn.utils import shuffle
from sklearn.metrics import log_loss

DEBUG: bool = False
VOCABULARY_PATH: str = "aclImdb/imdb.vocab"


class LogisticRegression:
    def __init__(self, h=0.0001, l=0.01, epochs=700):
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
        self.weights = np.random.randn(n_features).astype(
            np.float64
        )  # start with random weights

        for epoch in range(self.epochs):
            if DEBUG:
                total_loss = 0
                print(epoch)

            # Shuffle training examples
            X, y = shuffle(X, y, random_state=epoch)
            for i in range(n_data):
                # Compute prediction and loss for
                # current training example
                prediction = self._sigmoid(np.dot(self.weights, X[i]))
                error = y[i] - prediction
                gradient = error * X[i]

                # Calculate regularization
                regularization = np.sum(self.weights ** 2)
                regularization = np.clip(regularization, -1e-4, 1e-4)

                # Calculate weight update
                # based on gradient
                # and regularization
                weight_update = (self.h * gradient) - (self.l * regularization)
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
    preprocess = Preprocess(VOCABULARY_PATH)
    (
        x_train,
        y_train,
        x_dev,
        y_dev,
        x_test,
        y_test,
    ) = preprocess.preprocess_reviews()

    # Calculate accuracy in dev data
    # in order to determine hyperparameters
    lg = LogisticRegression(h=0.0001, l=0.01, epochs=700)
    train_sizes = [500, 1000, 3000, 5000, 10000, 15000, 20000, 25000]

    start = time.time()
    train_loss_scores, dev_loss_scores = [], []
    results = []
    for train_size in train_sizes:
        print(train_size)
        X = x_train[:train_size]
        y = y_train[:train_size]
        # Fit algorithm with a test dataset
        # the size of train_size
        lg.fit(X, y)

        # Calculate cross-entropy loss
        # on the training subset used
        train_pred = lg.predict(X)
        train_loss = log_loss(y_true=y, y_pred=train_pred)
        train_loss_scores.append(train_loss)

        # Calculate cross-entropy loss
        # on the dev dataset
        dev_pred = lg.predict(x_dev)
        dev_loss = log_loss(y_true=y_dev, y_pred=dev_pred)
        dev_loss_scores.append(dev_loss)

        results.append([train_size, round(train_loss, 2), round(dev_loss, 2)])
    end = time.time()

    columns = [
        "Train size",
        "Cross-entropy loss on training set",
        "Cross-entropy loss on dev set",
    ]
    table = pd.DataFrame(results, columns=columns)
    print(table)
    print(f"\nTotal rutime: {round(end - start, 3)} seconds.")

    _plot_learning_curve(train_sizes, train_loss_scores, dev_loss_scores)


def _plot_learning_curve(train_sizes, train_loss, dev_loss):
    plt.plot(train_sizes, train_loss, color="r", label="Training Set")
    plt.plot(train_sizes, dev_loss, color="g", label="Dev Set")

    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("Cross entropy loss")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
