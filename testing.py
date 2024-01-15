import pandas as pd
import matplotlib.pyplot as plt

from preprocess import Preprocess
from logistic_regression import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

TRAIN_SIZES: list = [500, 1000, 3000, 5000, 10000, 15000, 20000, 25000]


class Testing:
    def __init__(self, train_sizes: list = TRAIN_SIZES):
        self.train_sizes = train_sizes
        preprocess = Preprocess()
        (
            self.X_train,
            self.y_train,
            self.X_dev,
            self.y_dev,
            self.X_test,
            self.y_test,
        ) = preprocess.preprocess_reviews()

    def _print_table(self, score: str, results: list[list]) -> None:
        columns = ["Train size", f"{score} of training set", f"{score} of test set"]
        table = pd.DataFrame(results, columns=columns)
        print(table)
        print("\n")

    def _plot(
        self,
        train_sizes: list[int],
        train: list[float],
        test: list[float],
        ylabel: str,
        c1: str,
        c2: str,
    ) -> None:
        plt.plot(train_sizes, train, color=c1, label="Training Set")
        plt.plot(train_sizes, test, color=c2, label="Testing Set")

        plt.title("Learning Curve")
        plt.xlabel("Training Set Size")
        plt.ylabel(ylabel)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def evaluate_classifier(self, classifier: object) -> None:
        """
        Creating learning curves for accuracy, precision, recall and f1 score
        in train and test data, for various training sizes,
        in order to review the classifier.
        """
        train_accuracy_scores, test_accuracy_scores = [], []
        train_precision_scores, test_precision_scores = [], []
        train_recall_scores, test_recall_scores = [], []
        train_f1_scores, test_f1_scores = [], []
        accuracy_results, precision_results, recall_results, f1_results = [], [], [], []

        valid_train_sizes = []
        for train_size in self.train_sizes:
            # Check that train_size is in bounds
            if train_size >= len(self.X_train):
                continue

            valid_train_sizes.append(train_size)
            X = self.X_train[:train_size]
            y = self.y_train[:train_size]

            # Fit algorithm with a test dataset
            # the size of train_size
            classifier.fit(X, y)

            # Calculate metrics
            # on the training subset used
            train_pred = classifier.predict(X)
            train_accuracy = accuracy_score(y_true=y, y_pred=train_pred)
            train_accuracy_scores.append(train_accuracy)
            train_precision = precision_score(y_true=y, y_pred=train_pred)
            train_precision_scores.append(train_precision)
            train_recall = recall_score(y_true=y, y_pred=train_pred)
            train_recall_scores.append(train_recall)
            train_f1 = f1_score(
                y_true=y, y_pred=train_pred, pos_label=1, average="binary"
            )  # Returns the f1 score for the positive category
            train_f1_scores.append(train_f1)

            # Calculate metrics
            # on the testing dataset
            test_pred = classifier.predict(self.X_test)
            test_accuracy = accuracy_score(y_true=self.y_test, y_pred=test_pred)
            test_accuracy_scores.append(test_accuracy)
            test_precision = precision_score(y_true=self.y_test, y_pred=test_pred)
            test_precision_scores.append(test_precision)
            test_recall = recall_score(y_true=self.y_test, y_pred=test_pred)
            test_recall_scores.append(test_recall)
            test_f1 = f1_score(
                y_true=self.y_test, y_pred=test_pred, pos_label=1, average="binary"
            )  # Returns the f1 score for the positive category
            test_f1_scores.append(test_f1)

            accuracy_results.append(
                [train_size, round(train_accuracy, 2), round(test_accuracy, 2)]
            )
            precision_results.append(
                [train_size, round(train_precision, 2), round(test_precision, 2)]
            )
            recall_results.append(
                [train_size, round(train_recall, 2), round(test_recall, 2)]
            )
            f1_results.append([train_size, round(train_f1, 2), round(test_f1, 2)])

        self._print_table("Accuracy", accuracy_results)
        self._print_table("Precision", precision_results)
        self._print_table("Recall", recall_results)
        self._print_table("F1 Score", f1_results)

        # Plot accuracy
        self._plot(
            valid_train_sizes,
            train_accuracy_scores,
            test_accuracy_scores,
            ylabel="Accuracy Score",
            c1="r",
            c2="g",
        )
        # Plot precision
        self._plot(
            valid_train_sizes,
            train_precision_scores,
            test_precision_scores,
            ylabel="Precision Score",
            c1="c",
            c2="m",
        )
        # Plot recall
        self._plot(
            valid_train_sizes,
            train_recall_scores,
            test_recall_scores,
            ylabel="Recall Score",
            c1="g",
            c2="y",
        )
        # Plot F1 score
        self._plot(
            valid_train_sizes,
            train_f1_scores,
            test_f1_scores,
            ylabel="F1 Score",
            c1="b",
            c2="r",
        )


def main():
    testing = Testing()

    print("Logistic Regression:")
    testing.evaluate_classifier(LogisticRegression())


if __name__ == "__main__":
    main()
