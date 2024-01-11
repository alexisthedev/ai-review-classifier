import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocess import Preprocess
from decision_tree import DecisionTree
from sklearn.metrics import log_loss

VOCABULARY_PATH: str = (
    "C:/Users/serko/Desktop/sxoli/5ο εξαμηνο/τεχνητη νοημοσινη/aclImdb/imdb.vocab"
)


class RandomForest:
    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y) -> None:
        for _ in range(self.num_trees):
            # Randomly sample the data with replacement
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sampled = X[indices]
            y_sampled = y[indices]

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sampled, y_sampled)
            self.trees.append(tree)

    def predict(self, X) -> np.ndarray:
        predictions = [tree.predict(X) for tree in self.trees]
        ensemble_predictions = np.round(
            np.mean(predictions, axis=0)
        )  # Aggregate predictions by majority voting
        return np.round(ensemble_predictions)


def main() -> None:
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
    random_forest = RandomForest(num_trees=11, max_depth=10)
    train_sizes = [100, 1000, 5000, 10000, 15000, 20000, 25000]

    start = time.time()
    train_loss_scores, dev_loss_scores = [], []
    results = []
    for train_size in train_sizes:
        print(train_size)
        X = x_train[:train_size]
        y = y_train[:train_size]
        # Fit algorithm with a test dataset
        # the size of train_size
        random_forest.fit(X, y)

        # Calculate cross-entropy loss
        # on the training subset used
        train_pred = random_forest.predict(X)
        train_loss = log_loss(y_true=y, y_pred=train_pred)
        train_loss_scores.append(train_loss)

        # Calculate cross-entropy loss
        # on the dev dataset
        dev_pred = random_forest.predict(x_dev)
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
