import numpy as np
from decision_tree import DecisionTree


class RandomForest:
    def __init__(self, n_estimators=100, min_samples_split=10, max_depth=None, max_features=None, max_thresholds=200):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth, random_forest=True, max_thresholds=self.max_thresholds, max_features=self.max_features)
            tree.fit(X_sample, y_sample)
            if _ % 10 == 0: 
                print(f"Tree {_ + 1}/{self.n_estimators} trained.")
            self.trees.append(tree)

    def predict(self, X):
        tree_probabilities = np.array([tree.predict_proba(X) for tree in self.trees])
        mean_probabilities = np.mean(tree_probabilities, axis=0)
        return (mean_probabilities >= 0.5).astype(int)