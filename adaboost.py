import numpy as np

from sklearn.tree import DecisionTreeClassifier
from development import Development


class AdaBoost:
    def __init__(self, num_learners=800):
        self.num_learners = num_learners
        self.learner_weights = []
        self.learners = []

    def fit(self, X, y):
        self.learner_weights.clear()
        self.learners.clear()

        # Initialize weights
        weights = np.ones(len(X)) / len(X)

        for _ in range(self.num_learners):
            model = DecisionTreeClassifier(max_depth=1)

            # Train the weak learner
            model.fit(X, y, sample_weight=weights)

            predictions = model.predict(X)

            # Calculate weighted error
            weighted_error = np.sum(weights * (predictions != y))

            # Error normalization
            # A low weighted error leads to a higher alpha (learner performed well)
            # 1e-10 is used to prevent division by zero
            alpha = 0.5 * np.log((1 - weighted_error) / max(weighted_error, 1e-10))
            self.learner_weights.append(alpha)
            self.learners.append(model)

            # Decrease weight of correctly predicted samples
            for i in range(len(X)):
                if predictions[i] == y[i]:
                    weights[i] *= weighted_error / (1 - weighted_error)

            # Normalize weights so they sum up to 1
            weights /= weights.sum()

    def predict(self, X):
        # Make predictions using the ensemble of weak learners
        predictions = np.zeros(len(X))
        for weight, learner in zip(self.learner_weights, self.learners):
            predictions += weight * np.array([-1 if prediction == 0 else 1 for prediction in learner.predict(X)])
        return [0 if prediction < 0 else 1 for prediction in predictions]


def main():
    development = Development()

    development.evaluate_classifier(AdaBoost(num_learners=800))


if __name__ == "__main__":
    main()
