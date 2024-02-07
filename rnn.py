import time
import fasttext
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

VOCABULARY_PATH: str = "aclImdb/imdb.vocab"
DEVELOPMENT: bool = True
TESTING: bool = True
TRAIN_SIZES: list = [500, 1000, 3000, 5000, 10000, 15000, 20000, 25000]


class RNN:
    # Hyperparameters
    # Vocabulary
    N: int = 400  # Number of most common words to ignore
    K: int = 88500  # Number of least common words to ignore
    M: int = 0

    MAX_LENGTH: int = 250
    EMBEDDING_DIM: int = 300
    NUM_LAYERS: int = 1
    GRU_UNITS: int = 64
    EPOCHS: int = 10
    BATCHES: int = 32

    def __init__(self):
        # Import dataset
        (X_train_index, self.y_train), (X_test_index,y_test_imdb) = tf.keras.datasets.imdb.load_data()

        word_index = tf.keras.datasets.imdb.get_word_index()  # dict {word : index}
        index_to_word = dict(
            (i + 3, word) for (word, i) in word_index.items()
        )  # dict {index : word}
        # Add keywords
        index_to_word[0] = "[pad]"
        index_to_word[1] = "[bos]"
        index_to_word[2] = "[oov]"
        self.X_train = np.array(
            [" ".join([index_to_word[idx] for idx in text]) for text in X_train_index]
        )  # get string from indices
        X_test_string = np.array(
            [" ".join([index_to_word[idx] for idx in text]) for text in X_test_index]
        )  # get string from indices

        # Split test data to dev and test datasets
        self.X_dev, self.X_test, self.y_dev, self.y_test = train_test_split(
            X_test_string, y_test_imdb, test_size=6250, random_state=42
        )

        self.rnn = self.create_rnn()
        self.rnn.compile(loss="binary_crossentropy", optimizer="adam")

    def vectorizer_layer(self) -> object:
        """
        Creates and returns the vectorizer layer
        used in the RNN model to convert string reviews
        to vectors of integers.
        """
        # Process our vocabulary
        vocabulary = pd.read_fwf(
            VOCABULARY_PATH, skiprows=self.N, skipfooter=self.K, names=["vocab"]
        )
        self.M = vocabulary.size

        # Create vectorizer
        with tf.device("/CPU:0"):
            vectorizer = tf.keras.layers.TextVectorization(
                vocabulary=vocabulary.vocab,
                output_mode="int",
                name="vector_text",
                output_sequence_length=self.MAX_LENGTH,
            )

        return vectorizer

    def create_rnn(self) -> object:
        """
        Creates the RNN model.
        """
        # Input layer
        inputs = tf.keras.layers.Input(
            shape=(1,), dtype=tf.string, name="txt_input"
        )  # [string reviews]

        # Vectorizer layer
        vectorizer = self.vectorizer_layer()
        x = vectorizer(inputs)

        # Embedding layer
        fasttext_model = fasttext.load_model("cc.en.300.bin")
        embedding_matrix = np.zeros(shape=(len(vectorizer.get_vocabulary()), self.EMBEDDING_DIM))
        for i, word in enumerate(vectorizer.get_vocabulary()):
            embedding_matrix[i] = fasttext_model.get_word_vector(word=word)
        del fasttext_model

        x = tf.keras.layers.Embedding(
            input_dim=len(vectorizer.get_vocabulary()),
            output_dim=self.EMBEDDING_DIM,
            trainable=False,
            weights=[embedding_matrix],
            mask_zero=True,
            input_length=self.MAX_LENGTH,
        )(x)
        x = tf.keras.layers.Dropout(rate=0.25)(x)

        # RNN layers
        for n in range(self.NUM_LAYERS):
            if n != self.NUM_LAYERS - 1:
                x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=self.GRU_UNITS,
                                                name=f"bigru_cell_{n}",
                                                return_sequences=True,
                                                dropout=0.2))(x)
            else:
                x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                                                units=self.GRU_UNITS,
                                                name=f"bigru_cell_{n}",
                                                dropout=0.2))(x)

        x = tf.keras.layers.Dropout(rate=0.5)(x)
        o = tf.keras.layers.Dense(units=1, activation="sigmoid", name="lr")(x)  # binary classification

        return tf.keras.models.Model(inputs=inputs, outputs=o, name="bigru_rnn")

    def _plot_loss(self, epochs: list, train_loss: list, dev_loss: list) -> None:
        plt.plot(epochs, train_loss, color="r", label="Training Set")
        plt.plot(epochs, dev_loss, color="g", label="Dev Set")

        plt.title("Learning Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def fit(self, X, y, verbose: int):
        history = self.rnn.fit(
            x=X,
            y=y,
            epochs=self.EPOCHS,
            batch_size=self.BATCHES,
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
