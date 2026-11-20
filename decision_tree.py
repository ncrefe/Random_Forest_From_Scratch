import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, min_samples_split=10, max_depth=None, random_forest=False, max_features=None, max_thresholds=200):
        self.root = None
        self.minimum_samples_split = min_samples_split
        self.max_depth = max_depth
        self.random_forest = random_forest
        self.max_features = max_features
        self.max_thresholds = max_thresholds

        ### statistical variables for termination conditions
        self.termination_by_depth = 0
        self.termination_by_samples = 0
        self.termination_by_purity = 0

    def fit(self, X, y):
        self.root = self._build_tree(X, y, 0)

    def _build_tree(self, X, y, depth):
        if X.shape[0] < self.minimum_samples_split:
            ### holds probability of class 1 for classification
            self.termination_by_samples += 1
            return Node(value=np.mean(y))

        if len(np.unique(y)) == 1:
            self.termination_by_purity += 1
            return Node(value=y[0])
        
        if self.max_depth is not None and depth >= self.max_depth:
            self.termination_by_depth += 1
            return Node(value=np.mean(y))

        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return Node(value=np.mean(y))

        # Split the data
        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices

        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _find_best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_info_gain = 0

        parent_shannon_entropy = self.calculate_shannon_entropy(y)

        if self.random_forest:
            if self.max_features is None:
                number_features = int(np.sqrt(X.shape[1]))
            else:
                number_features = self.max_features

            features = np.random.choice(X.shape[1], size=number_features, replace=False)
        else:
            features = range(X.shape[1])

        for feature in features:
            thresholds = self.obtain_thresholds(X, feature)
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = ~left_indices

                if np.any(left_indices) and np.any(right_indices):
                    information_gain = self.calculate_information_gain(parent_shannon_entropy, y[left_indices], y[right_indices])
                    if information_gain > best_info_gain:
                        best_info_gain = information_gain
                        best_feature = feature
                        best_threshold = threshold

        return best_feature, best_threshold

    def obtain_thresholds(self, X, feature):
        unique_values = np.unique(X[:, feature])

        if len(unique_values) <= 1:
            return unique_values

        unique_values = unique_values[1:]

        if len(unique_values) <= self.max_thresholds:
            return unique_values

        sampled_thresholds = np.random.choice(
            unique_values,
            size=self.max_thresholds,
            replace=False
        )

        return sampled_thresholds

    def calculate_information_gain(self, parent_entropy, left_y, right_y):
        left_entropy = self.calculate_shannon_entropy(left_y)
        right_entropy = self.calculate_shannon_entropy(right_y)

        total_samples = len(left_y) + len(right_y)

        weight_left = len(left_y) / total_samples
        weight_right = len(right_y) / total_samples

        return parent_entropy - (weight_left * left_entropy + weight_right * right_entropy)


    def calculate_shannon_entropy(self, y):
       ### 0 and 1 for binary classification
       p_class1 = np.mean(y)
       p_class0 = 1 - p_class1
       
       if p_class1 == 0 or p_class1 == 1:
           return 0
       return -p_class1 * np.log2(p_class1) - (p_class0) * np.log2(p_class0)


    def predict_proba(self, X):
        probabilities = []
        for x in X:
            node = self.root

            while node.left is not None or node.right is not None:

                if x[node.feature] < node.threshold:
                    node = node.left
                else:
                    node = node.right

            probabilities.append(node.value)
        return np.array(probabilities)
    
    def predict(self, X):
        predictions = []
        for x in X:
            node = self.root

            while node.left is not None or node.right is not None:

                if x[node.feature] < node.threshold:
                    node = node.left
                else:
                    node = node.right

            prediction = 1 if node.value >= 0.5 else 0
            predictions.append(prediction)
        return np.array(predictions)

    def print_termination_stats(self):
        print(f"Termination by depth: {self.termination_by_depth}")
        print(f"Termination by samples: {self.termination_by_samples}")
        print(f"Termination by purity: {self.termination_by_purity}")