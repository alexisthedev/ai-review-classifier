import pandas as pd
from pathlib import Path
import time

POSITIVE: int = 1
NEGATIVE: int = 0
VOCABULARY_PATH: str = 'aclImdb/imdb.vocab'
TRAINING_REVIEW_PATH: str = 'aclImdb/train/'

class Preprocess:
    # Hyperparameters
    N: int = 400 # Number of most common words to ignore
    K: int = 88500 # Number of least common words to ignore
    M: int = 0

    def __init__(self, vocabulary_path: str) -> None:
        self.vocabulary = self.extract_vocabulary(vocabulary_path)

    def extract_vocabulary(self, vocabulary_path: str) -> dict[str, int]:
        '''
        Parses dataset vocabulary and creates dictionary mapping
        each word to its index in a review's attribute vector.
        '''
        vocabulary = pd.read_fwf(vocabulary_path, skiprows=self.N, skipfooter=self.K, names=['vocab'])
        self.M = vocabulary.size
        return dict(zip(vocabulary.vocab, range(self.M)))

    def preprocess_review(self, review_path: str) -> list[int]:
        '''
        Parses review txt file and creates binary vector.
        '''
        review = pd.read_fwf(review_path, names=['review'])
        review_words = list(review.iloc[0].name)
        vector = [0]*self.M
        for word in review_words:
            if word in self.vocabulary:
                vector[self.vocabulary.get(word)] = 1
        return vector

    def preprocess_reviews(self, pos_path='', neg_path='') -> list[tuple[list, int]]:
        '''
        Iterates through positive / negative reviews directory,
        processes every review and returns a list
        containg tuples with the vector and
        the expected outcome for each review .
        '''
        res = []
        path = f'{TRAINING_REVIEW_PATH}{pos_path}' if pos_path else f'{TRAINING_REVIEW_PATH}{neg_path}'
        reviews = Path(path).glob('*.txt')
        for review in reviews:
            vector = self.preprocess_review(review)
            outcome = POSITIVE * (len(pos_path) != 0)
            res.append((vector, outcome))
        return res

def main():
    preprocess = Preprocess(VOCABULARY_PATH)
    start_pos = time.time()
    positive_reviews = preprocess.preprocess_reviews(pos_path='pos/')
    end_pos = time.time()
    start_neg = time.time()
    negative_reviews = preprocess.preprocess_reviews(neg_path='neg/')
    end_neg = time.time()

    print(f'Positive reviews: {len(positive_reviews)}\nCalculated in {round(end_pos - start_pos, 3)} seconds')
    print(f'Negative reviews: {len(negative_reviews)}\nCalculated in {round(end_neg - start_neg, 3)} seconds')

if __name__ == '__main__':
    main()
