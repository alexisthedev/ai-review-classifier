import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

TRAIN_SIZES: list = [500, 1000, 3000, 5000, 10000, 15000, 20000, 25000]

class RNN:
    # Hyperparameters
    # Vocabulary
    N: int = 400  # Number of most common words to ignore
    K: int = 88500  # Number of least common words to ignore
    M: int = 0

    EMBEDDING_DIM: int = 60
    EPOCHS: int = 10

    def __init__(self):
        word_index = tf.keras.datasets.imdb.get_word_index() # dict {word : index}
        total_vocabulary_size = len(word_index)
        self.M = total_vocabulary_size - self.N - self.K

        # TODO add vocabulary restrictions
        (X_train_imdb, self.y_train), (X_test_imdb, y_test_imdb) = tf.keras.datasets.imdb.load_data()

        # Split test data to dev and test datasets
        X_dev_imdb, X_test_imdb, self.y_dev, self.y_test = train_test_split(
            X_test_imdb, y_test_imdb, test_size=6250, random_state=42
        )

        # Create tensors
        temp_X_train = [tf.constant(lst) for lst in X_train_imdb]
        self.X_train = tf.ragged.stack(temp_X_train)
        temp_X_dev = [tf.constant(lst) for lst in X_dev_imdb]
        self.X_dev = tf.ragged.stack(temp_X_dev)
        temp_X_test = [tf.constant(lst) for lst in X_test_imdb]
        self.X_test = tf.ragged.stack(temp_X_test)

        self.rnn = tf.keras.models.Sequential()
        self.rnn.add(tf.keras.layers.Embedding(input_dim=total_vocabulary_size + 3, output_dim=self.EMBEDDING_DIM))
        self.rnn.add(tf.keras.layers.GRU(units=28, activation="tanh", dropout=0.5, recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        self.rnn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))  # binary classification

        self.rnn.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.legacy.Adam())

    def _plot_loss(
        self, epochs: list, train_loss: list, dev_loss: list
    ) -> None:
        plt.plot(epochs, train_loss, color="r", label="Training Set")
        plt.plot(epochs, dev_loss, color="g", label="Dev Set")

        plt.title("Learning Curve")
        plt.xlabel("Training Set Size")
        plt.ylabel("Cross-entropy loss")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def _print_table(self, score: str, results: list[list]) -> None:
        columns = ["Train size", f"{score} of training set", f"{score} of test set"]
        table = pd.DataFrame(results, columns=columns)
        print(table)
        print("\n")

    def _plot_learning_curve(
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

    def fit(self, verbose: int):
        history = self.rnn.fit(x=self.X_train, y=self.y_train, epochs=self.EPOCHS, validation_data=(self.X_dev, self.y_dev), verbose=verbose)

        if verbose == 1:
            self._plot_loss([x + 1 for x in range(RNN.EPOCHS)], history.history['loss'], history.history['val_loss'])

    def evaluate_classifier(self, train_sizes: list = TRAIN_SIZES) -> None:
            """
            Creating learning curves for accuracy, precision, recall and f1 score
            in train and test data, for various training sizes,
            in order to review the classifier.
            """
            classifier = RNN()

            train_accuracy_scores, test_accuracy_scores = [], []
            train_precision_scores, test_precision_scores = [], []
            train_recall_scores, test_recall_scores = [], []
            train_f1_scores, test_f1_scores = [], []
            accuracy_results, precision_results, recall_results, f1_results = [], [], [], []

            for train_size in train_sizes:
                X = self.X_train[:train_size]
                y = self.y_train[:train_size]

                # Fit algorithm with a test dataset
                # the size of train_size
                classifier.fit(X, y, verbose=0)

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
            self._plot_learning_curve(
                train_sizes,
                train_accuracy_scores,
                test_accuracy_scores,
                ylabel="Accuracy Score",
                c1="r",
                c2="g",
            )
            # Plot precision
            self._plot_learning_curve(
                train_sizes,
                train_precision_scores,
                test_precision_scores,
                ylabel="Precision Score",
                c1="c",
                c2="m",
            )
            # Plot recall
            self._plot_learning_curve(
                train_sizes,
                train_recall_scores,
                test_recall_scores,
                ylabel="Recall Score",
                c1="g",
                c2="y",
            )
            # Plot F1 score
            self._plot_learning_curve(
                train_sizes,
                train_f1_scores,
                test_f1_scores,
                ylabel="F1 Score",
                c1="b",
                c2="r",
            )


def main():
    rnn = RNN()

    # Development
    rnn.fit(verbose=1)

    # Testing
    rnn.evaluate_classifier()


if __name__ == "__main__":
    main()
