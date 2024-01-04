import numpy as np


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def train(self, X, y, depth=0):
        # ID3 training recursive function
        if depth == self.max_depth or len(set(y)) == 1:
            self.tree = {"label": np.argmax(np.bincount(y))}
            return self.tree

        num_features = X.shape[1]
        best_feature, best_threshold = None, None
        best_gini = float("inf")

        for feature_index in range(num_features):
            unique_values = np.unique(X[:, feature_index])
            for value in unique_values:
                left_indices = X[:, feature_index] <= value
                right_indices = ~left_indices
                gini = self.calculate_gini(y[left_indices], y[right_indices])

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = value

        if best_gini == float("inf"):
            self.tree = {"label": max(set(y), key=y.count)}
            return self.tree

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices

        # Recursively build the left and right subtrees
        left_subtree = self.train(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self.train(X[right_indices], y[right_indices], depth + 1)

        # Create a node representing the best split
        self.tree = {
            "feature_index": best_feature,
            "threshold": best_threshold,
            "left": left_subtree,
            "right": right_subtree,
        }
        return self.tree

    def predict(self, X):
        predictions = np.zeros(len(X))
        for i in range(len(X)):
            predictions[i] = self._predict_single(X[i], self.tree)
        return predictions

    def _predict_single(self, example, node):
        # ID3 prediction recursive function
        if "label" in node:
            # Reached a leaf node, return the label
            return node["label"]

        # Check the feature of the current node
        feature_index = node["feature_index"]
        threshold = node["threshold"]

        # Move to the left or right subtree based on the input feature
        if example[feature_index] <= threshold:
            return self._predict_single(example, node["left"])
        else:
            return self._predict_single(example, node["right"])

    def calculate_gini(self, left_labels, right_labels):
        total_samples = len(left_labels) + len(right_labels)
        gini_left = 1.0 - sum(
            (np.sum(left_labels == label) / len(left_labels)) ** 2
            for label in np.unique(left_labels)
        )
        gini_right = 1.0 - sum(
            (np.sum(right_labels == label) / len(right_labels)) ** 2
            for label in np.unique(right_labels)
        )
        weighted_gini = (len(left_labels) / total_samples) * gini_left + (
            len(right_labels) / total_samples
        ) * gini_right
        return weighted_gini
