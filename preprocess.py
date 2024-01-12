import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

VOCABULARY_PATH: str = "aclImdb/imdb.vocab"
TRAINING_REVIEW_PATH: str = "aclImdb/train/"


class Preprocess:
    # Hyperparameters
    N: int = 400  # Number of most common words to ignore
    K: int = 88500  # Number of least common words to ignore
    M: int = 0

    def __init__(self) -> None:
        self.vocabulary = self.extract_vocabulary(VOCABULARY_PATH)
        self.vectorizer = CountVectorizer(vocabulary=self.vocabulary, binary=True)

    def extract_vocabulary(self, vocabulary_path: str) -> dict[str, int]:
        """
        Parses dataset vocabulary and creates dictionary mapping
        each word to its index in a review's attribute vector.
        """
        vocabulary = pd.read_fwf(
            vocabulary_path, skiprows=self.N, skipfooter=self.K, names=["vocab"]
        )
        self.M = vocabulary.size
        return dict(zip(vocabulary.vocab, range(self.M)))

    def preprocess_reviews(self):
        (x_train_imdb, y_train_imdb), (x_test_imdb, y_test_imdb) = tf.keras.datasets.imdb.load_data()

        word_index = tf.keras.datasets.imdb.get_word_index()  # dict {word : index}
        index_to_word = dict(
            (i + 3, word) for (word, i) in word_index.items()
        )  # dict {index : word}
        # Add keywords
        index_to_word[0] = "[pad]"
        index_to_word[1] = "[bos]"
        index_to_word[2] = "[oov]"
        x_train_imdb = np.array(
            [" ".join([index_to_word[idx] for idx in text]) for text in x_train_imdb]
        )  # get string from indices
        x_test_imdb = np.array(
            [" ".join([index_to_word[idx] for idx in text]) for text in x_test_imdb]
        )  # get string from indices

        # Create vectors
        x_train_imdb_binary = self.vectorizer.transform(x_train_imdb).toarray()
        x_test_imdb_binary = self.vectorizer.transform(x_test_imdb).toarray()

        # Split test data to dev and test datasets
        x_dev, x_test, y_dev, y_test = train_test_split(
            x_test_imdb_binary, y_test_imdb, test_size=6250, random_state=42
        )

        return (x_train_imdb_binary, y_train_imdb, x_dev, y_dev, x_test, y_test)


def main():
    preprocess = Preprocess()
    (
        x_train,
        y_train,
        x_dev,
        y_dev,
        x_test,
        y_test
    ) = preprocess.preprocess_reviews()


if __name__ == "__main__":
    main()
