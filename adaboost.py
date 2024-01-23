import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
from preprocess import Preprocess

VOCABULARY_PATH: str = "aclImdb/imdb.vocab"

class AdaBoost:
    def __init__(self, num_learners=800):
        self.num_learners = num_learners
        self.learner_weights = []
        self.learners = []

    def fit(self, X, y):
        self.learner_weights.clear()
        self.learners.clear()

        # Initialize weights
        weights = np.ones(len(X)) / len(X)

        for _ in range(self.num_learners):
            model = DecisionTreeClassifier(max_depth=1)

            # Train the weak learner
            model.fit(X, y, sample_weight=weights)

            predictions = model.predict(X)

            # Calculate weighted error
            weighted_error = np.sum(weights * (predictions != y))

            # Error normalization
            # A low weighted error leads to a higher alpha (learner performed well)
            # 1e-10 is used to prevent division by zero
            alpha = 0.5 * np.log((1 - weighted_error) / max(weighted_error, 1e-10))
            self.learner_weights.append(alpha)
            self.learners.append(model)

            # Decrease weight of correctly predicted samples
            for i in range(len(X)):
                if predictions[i] == y[i]:
                    weights[i] *= weighted_error / (1 - weighted_error)

            # Normalize weights so they sum up to 1
            weights /= weights.sum()

    def predict(self, X):
        # Make predictions using the ensemble of weak learners
        predictions = np.zeros(len(X))
        for weight, learner in zip(self.learner_weights, self.learners):
            predictions += weight * np.array([-1 if prediction == 0 else 1 for prediction in learner.predict(X)])
        return [0 if prediction < 0 else 1 for prediction in predictions]

def main():
    # Load the preprocessed IMDb dataset
    preprocess = Preprocess(VOCABULARY_PATH)
    (
        X_train,
        y_train,
        X_dev,
        y_dev,
        X_test,
        y_test
    ) = preprocess.preprocess_reviews()

    # Calculate cross-entropy loss in dev data
    # in order to determine hyperparameters
    adaboost = AdaBoost(800)
    train_sizes = [500, 1000, 3000, 5000, 10000, 15000, 20000, 25000]

    start = time.time()
    train_loss_scores, dev_loss_scores = [], []
    results = []
    for train_size in train_sizes:
        print(train_size)
        X = X_train[:train_size]
        y = y_train[:train_size]

        # Fit algorithm with train subset
        adaboost.fit(X, y)

        # Calculate cross-entropy loss for train subset
        y_train_pred = adaboost.predict(X)
        train_loss = log_loss(y_true=y, y_pred=y_train_pred)
        train_loss_scores.append(train_loss)

        # Calculate cross-entropy loss on dev
        y_dev_pred = adaboost.predict(X_dev)
        dev_loss = log_loss(y_true=y_dev, y_pred=y_dev_pred)
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
    print(f"\nTotal runtime: {round(end - start, 3)} seconds.")

    _plot_learning_curve(train_sizes, train_loss_scores, dev_loss_scores)


def _plot_learning_curve(train_sizes, train_loss, dev_loss):
    plt.plot(train_sizes, train_loss, color="r", label="Training Set")
    plt.plot(train_sizes, dev_loss, color="g", label="Dev Set")

    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("Cross-entropy loss")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
