import numpy as np
from Preprocess import Preprocess
from decision_tree import DecisionTree
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import time

VOCABULARY_PATH: str = "C:/Users/serko/Desktop/sxoli/5ο εξαμηνο/τεχνητη νοημοσινη/aclImdb/imdb.vocab"

class RandomForest:
    def __init__(self, num_trees):
        self.num_trees = num_trees
        self.trees = []

    def train(self, X, y):
        for _ in range(self.num_trees):
            # Randomly sample the data with replacement
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sampled = X[indices]
            y_sampled = y[indices]

            tree = DecisionTree(max_depth=3)
            tree.train(X_sampled, y_sampled)
            self.trees.append(tree)

    def predict(self, X):
        predictions = [tree.predict(X) for tree in self.trees]
        ensemble_predictions = np.round(np.mean(predictions, axis=0)) # Aggregate predictions by majority voting
        return np.round(ensemble_predictions)

def main():
    start_time = time.time()
    preprocess = Preprocess(VOCABULARY_PATH)
    x_train_imdb_binary, y_train_imdb, x_dev, y_dev, x_test_imdb_binary, y_test_imdb = preprocess.preprocess_reviews()

    random_forest = RandomForest(num_trees=10)
    random_forest.train(x_train_imdb_binary, y_train_imdb)

    predictions = random_forest.predict(x_test_imdb_binary)

    # Evaluate the accuracy
    accuracy = accuracy_score(y_test_imdb, predictions)
    print(f"Random Forest Accuracy: {accuracy}")
    end_time = time.time()

    # Calculate and print the runtime
    runtime = end_time - start_time
    print(f"Total runtime: {runtime} seconds")

if __name__ == '__main__':
    main()
