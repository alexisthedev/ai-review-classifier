import numpy as np

from decision_tree import DecisionTree
from development import Development


class RandomForest:
    def __init__(self, num_trees=11, max_depth=10):
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
    development = Development()

    development.evaluate_classifier(RandomForest(num_trees=11, max_depth=10))


if __name__ == "__main__":
    main()
