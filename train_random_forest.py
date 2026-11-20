import numpy as np
import random
import os
import openml
from random_forest import RandomForest



def preprocess_features(X_train, X_test):

    X_train = X_train.copy()
    X_test = X_test.copy()

    # ===== Feature 0: Airline =====
    airlines = np.unique(X_train[:, 0])

    airline_to_int = {
        airline: idx
        for idx, airline in enumerate(airlines)
    }

    X_train[:, 0] = [
        airline_to_int[x]
        for x in X_train[:, 0]
    ]

    X_test[:, 0] = [
        airline_to_int.get(x, -1)
        for x in X_test[:, 0]
    ]

    # ===== Feature 1: Flight Number =====
    X_train[:, 1] = X_train[:, 1].astype(float)
    X_test[:, 1] = X_test[:, 1].astype(float)

    # ===== Feature 2: Airport From =====
    airports_from = np.unique(X_train[:, 2])

    airport_from_to_int = {
        airport: idx
        for idx, airport in enumerate(airports_from)
    }

    X_train[:, 2] = [
        airport_from_to_int[x]
        for x in X_train[:, 2]
    ]

    X_test[:, 2] = [
        airport_from_to_int.get(x, -1)
        for x in X_test[:, 2]
    ]

    # ===== Feature 3: Airport To =====
    airports_to = np.unique(X_train[:, 3])

    airport_to_to_int = {
        airport: idx
        for idx, airport in enumerate(airports_to)
    }

    X_train[:, 3] = [
        airport_to_to_int[x]
        for x in X_train[:, 3]
    ]

    X_test[:, 3] = [
        airport_to_to_int.get(x, -1)
        for x in X_test[:, 3]
    ]

    # ===== Remaining numeric features =====
    X_train[:, 4:] = X_train[:, 4:].astype(float)
    X_test[:, 4:] = X_test[:, 4:].astype(float)

    # Final conversion to float matrix
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    return X_train, X_test




seed = 10

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)


air = openml.datasets.get_dataset(42493)

X_air, y_air, _, _ = air.get_data(
    target=air.default_target_attribute
)

# Convert to NumPy arrays
X_air = np.array(X_air)
y_air = np.array(y_air)


n = X_air.shape[0]

indices = np.random.permutation(n)

split = int(0.8 * n)

train_idx = indices[:split]
test_idx = indices[split:]

X_train = X_air[train_idx]
X_test = X_air[test_idx]

y_train = y_air[train_idx]
y_test = y_air[test_idx]



X_train, X_test = preprocess_features(X_train, X_test)
y_train = y_train.astype(int)
y_test = y_test.astype(int)



n_estimators = 200
min_samples_split = 400
max_depth = 6
max_features = 6
max_thresholds = 200

print("Number of estimators:", n_estimators)
print("Min samples split:", min_samples_split)
print("Max depth:", max_depth)
print("Max features:", max_features)
print("Max thresholds:", max_thresholds)

tree = RandomForest(n_estimators=n_estimators, min_samples_split=min_samples_split, max_depth=max_depth, max_features=max_features, max_thresholds=max_thresholds)

print("\nTraining Random Forest...")

tree.fit(X_train, y_train)

print("Training finished.")

print("\nRunning predictions...")

predictions = tree.predict(X_test)

accuracy = np.mean(predictions == y_test)

print("\nTest Accuracy:", accuracy)
