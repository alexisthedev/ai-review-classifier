import time
import pandas as pd
import matplotlib.pyplot as plt

from preprocess import Preprocess
from sklearn.metrics import log_loss

TRAIN_SIZES: list = [500, 1000, 3000, 5000, 10000, 15000, 20000, 25000]
COLUMNS: list = [
    "Train size",
    "Cross-entropy loss on training set",
    "Cross-entropy loss on dev set",
]


class Development:
    def __init__(self, train_sizes: list = TRAIN_SIZES):
        preprocess = Preprocess()
        (
            self.X_train,
            self.y_train,
            self.X_dev,
            self.y_dev,
            self.X_test,
            self.y_test,
        ) = preprocess.preprocess_reviews()
        self.train_sizes = train_sizes

    def _plot_learning_curve(
        self, train_sizes: list, train_loss: list, dev_loss: list
    ) -> None:
        plt.plot(train_sizes, train_loss, color="r", label="Training Set")
        plt.plot(train_sizes, dev_loss, color="g", label="Dev Set")

        plt.title("Learning Curve")
        plt.xlabel("Training Set Size")
        plt.ylabel("Cross-entropy loss")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def evaluate_classifier(self, classifier: object) -> None:
        """
        Calculate cross-entropy loss
        in train and dev data, for various training sizes,
        in order to determine hyperparameters.
        """
        start = time.time()
        valid_train_sizes = []
        train_loss_scores, dev_loss_scores = [], []
        results = []
        for train_size in self.train_sizes:
            # Check that train_size is in bounds
            if train_size > len(self.X_train):
                continue

            valid_train_sizes.append(train_size)
            print(train_size)
            X = self.X_train[:train_size]
            y = self.y_train[:train_size]

            # Fit classifier with a test dataset
            # the size of train_size
            classifier.fit(X, y)

            # Calculate cross-entropy loss
            # on the training subset used
            train_pred = classifier.predict(X)
            train_loss = log_loss(y_true=y, y_pred=train_pred)
            train_loss_scores.append(train_loss)

            # Calculate cross-entropy loss
            # on the dev dataset
            dev_pred = classifier.predict(self.X_dev)
            dev_loss = log_loss(y_true=self.y_dev, y_pred=dev_pred)
            dev_loss_scores.append(dev_loss)

            results.append([train_size, round(train_loss, 2), round(dev_loss, 2)])
        end = time.time()

        table = pd.DataFrame(results, columns=COLUMNS)
        print(table)
        print(f"\nTotal runtime: {round(end - start, 3)} seconds.")

        self._plot_learning_curve(valid_train_sizes, train_loss_scores, dev_loss_scores)
