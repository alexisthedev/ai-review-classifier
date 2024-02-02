import time
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DEVELOPMENT: bool = True
TESTING: bool = False
TRAIN_SIZES: list = [500, 1000, 3000, 5000, 10000, 15000, 20000, 25000]


class RNN:
    # Hyperparameters
    # Vocabulary
    N: int = 400  # Number of most common words to ignore
    M: int = 627  # Words to use in vocabulary

    EMBEDDING_DIM: int = 64
    LSTM_UNITS: int = 100
    EPOCHS: int = 10

    def __init__(self):
        # Import vocabulary
        word_index = tf.keras.datasets.imdb.get_word_index()  # dict {word : index}
        total_vocabulary_size = len(word_index)

        # Import dataset
        (self.X_train, self.y_train), (X_test_imdb, y_test_imdb) = tf.keras.datasets.imdb.load_data(num_words=self.N + self.M, skip_top=self.N)

        # Pad the train and test lists
        max_length = 400
        self.X_train = sequence.pad_sequences(self.X_train, maxlen=max_length)
        X_test_imdb = sequence.pad_sequences(X_test_imdb, maxlen=max_length)

        # Split test data to dev and test datasets
        self.X_dev, self.X_test, self.y_dev, self.y_test = train_test_split(
            X_test_imdb, y_test_imdb, test_size=6250, random_state=42
        )

        # Create RNN model
        self.rnn = tf.keras.models.Sequential()
        self.rnn.add(
            tf.keras.layers.Embedding(
                input_dim=total_vocabulary_size + 3,
                output_dim=self.EMBEDDING_DIM,
                input_length=max_length,
            )
        )
        self.rnn.add(tf.keras.layers.Dropout(0.1))
        self.rnn.add(
            tf.keras.layers.LSTM(
                units=self.LSTM_UNITS,
                activation="tanh",
            )
        )
        self.rnn.add(tf.keras.layers.Dropout(0.3))
        self.rnn.add(
            tf.keras.layers.Dense(units=1, activation="sigmoid")
        )  # binary classification

        self.rnn.compile(loss="binary_crossentropy", optimizer="adam")

    def _plot_loss(self, epochs: list, train_loss: list, dev_loss: list) -> None:
        plt.plot(epochs, train_loss, color="r", label="Training Set")
        plt.plot(epochs, dev_loss, color="g", label="Dev Set")

        plt.title("Learning Curve")
        plt.xlabel("Training Set Size")
        plt.ylabel("Cross-entropy loss")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def fit(self, X, y, verbose: int):
        history = self.rnn.fit(
            x=X,
            y=y,
            epochs=self.EPOCHS,
            validation_data=(self.X_dev, self.y_dev),
            verbose=verbose,
        )

        if verbose == 1:
            self._plot_loss(
                [x + 1 for x in range(self.EPOCHS)],
                history.history["loss"],
                history.history["val_loss"],
            )

    def predict(self, X):
        predictions = self.rnn.predict(x=X, verbose=0)
        binary_predictions = [0 if pred < 0.5 else 1 for pred in predictions]
        return binary_predictions


def _print_table(score: str, results: list[list]) -> None:
    columns = ["Train size", f"{score} of training set", f"{score} of test set"]
    table = pd.DataFrame(results, columns=columns)
    print(table)
    print("\n")

def _plot_learning_curve(
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

def development() -> None:
    classifier = RNN()
    classifier.fit(classifier.X_train, classifier.y_train, verbose=1)

def evaluate_classifier() -> None:
    """
    Creating learning curves for accuracy, precision, recall and f1 score
    in train and test data, for various training sizes,
    in order to review the classifier.
    """
    print("\nTesting...")
    classifier = RNN()

    train_accuracy_scores, test_accuracy_scores = [], []
    train_precision_scores, test_precision_scores = [], []
    train_recall_scores, test_recall_scores = [], []
    train_f1_scores, test_f1_scores = [], []
    accuracy_results, precision_results, recall_results, f1_results = [], [], [], []

    for train_size in TRAIN_SIZES:
        print(f"Fitting with {train_size} samples...")
        X = classifier.X_train[:train_size]
        y = classifier.y_train[:train_size]

        # Fit algorithm with a test dataset
        # the size of train_size
        start = time.time()
        classifier.fit(X, y, verbose=0)
        end = time.time()
        print(
            f"Fitting completed in {round(end - start, 2)} seconds. Starting training calculations..."
        )

        # Calculate metrics
        # on the training subset used
        train_pred = classifier.predict(X)
        print(train_pred)
        print(train_pred[0])
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
        print(f"Training calculations completed. Starting testing calculations...")

        # Calculate metrics
        # on the testing dataset
        test_pred = classifier.predict(classifier.X_test)
        test_accuracy = accuracy_score(y_true=classifier.y_test, y_pred=test_pred)
        test_accuracy_scores.append(test_accuracy)
        test_precision = precision_score(y_true=classifier.y_test, y_pred=test_pred)
        test_precision_scores.append(test_precision)
        test_recall = recall_score(y_true=classifier.y_test, y_pred=test_pred)
        test_recall_scores.append(test_recall)
        test_f1 = f1_score(
            y_true=classifier.y_test, y_pred=test_pred, pos_label=1, average="binary"
        )  # Returns the f1 score for the positive category
        test_f1_scores.append(test_f1)
        print(f"Testing calculations completed. Wrapping up current train size...")

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

    _print_table("Accuracy", accuracy_results)
    _print_table("Precision", precision_results)
    _print_table("Recall", recall_results)
    _print_table("F1 Score", f1_results)

    # Plot accuracy
    _plot_learning_curve(
        TRAIN_SIZES,
        train_accuracy_scores,
        test_accuracy_scores,
        ylabel="Accuracy Score",
        c1="r",
        c2="g",
    )
    # Plot precision
    _plot_learning_curve(
        TRAIN_SIZES,
        train_precision_scores,
        test_precision_scores,
        ylabel="Precision Score",
        c1="c",
        c2="m",
    )
    # Plot recall
    _plot_learning_curve(
        TRAIN_SIZES,
        train_recall_scores,
        test_recall_scores,
        ylabel="Recall Score",
        c1="g",
        c2="y",
    )
    # Plot F1 score
    _plot_learning_curve(
        TRAIN_SIZES,
        train_f1_scores,
        test_f1_scores,
        ylabel="F1 Score",
        c1="b",
        c2="r",
    )

def main():
    if DEVELOPMENT:
        development()

    if TESTING:
        evaluate_classifier()


if __name__ == "__main__":
    main()
