# ✈️ Decision Tree & Random Forest from Scratch (NumPy): Flight Delay Prediction

This project implements a **decision tree classifier** and a **random forest** using **only NumPy**, and applies them to a real-world tabular problem: predicting whether a flight will be **delayed** from its schedule and route information.

The focus is on understanding how tree-based models work internally (entropy, information gain, threshold search, bootstrapping, feature subsampling) and on how hyperparameters and randomness affect generalisation.

---

## 🚀 Objectives

- Implement a decision tree with **information-gain** splits
- Handle **mixed categorical and numerical features**, including high-cardinality ones
- Tune `max_depth` and `min_samples_split` to control overfitting
- Build a **random forest** with bootstrap sampling and per-split feature subsampling
- Compare a single tree against the ensemble
- Keep every experiment reproducible with fixed seeds and a fixed split

---

## 📂 Project Structure

```
.
├── decision_tree.py          # Node + DecisionTree (entropy, info gain, thresholds)
├── random_forest.py          # RandomForest (bootstrap + feature subsampling)
├── train_decision_tree.py    # data loading, preprocessing, training, evaluation
├── train_random_forest.py    # same pipeline with the forest
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

The **Airlines** dataset from OpenML (id `42493`) has roughly 27,000 flights and 8 input features: 6 categorical-style (airline, flight number, origin airport, destination airport, ...) and 2 numerical (e.g. departure time, length). The target is binary: delayed or not.

**Split and reproducibility:**

```python
seed = 10
np.random.seed(seed)
indices = np.random.permutation(n)
split = int(0.8 * n)             # 80% train / 20% test
```

---

## 🧠 Method Overview

### 1. Preprocessing

Trees need numeric input, so the categorical columns are **label-encoded using the training set only**:

```python
airlines = np.unique(X_train[:, 0])
airline_to_int = {a: i for i, a in enumerate(airlines)}
X_train[:, 0] = [airline_to_int[x] for x in X_train[:, 0]]
X_test[:, 0]  = [airline_to_int.get(x, -1) for x in X_test[:, 0]]   # unseen → -1
```

- High-cardinality columns (airline, origin, destination) become ordinal codes. Threshold splits on them work as grouping of codes, and no one-hot explosion is needed.
- Categories never seen in training map to `-1`, so test data cannot leak into the encoding.
- The remaining columns are cast to float.

### 2. Decision Tree (`decision_tree.py`)

At each node the tree searches for the split with the highest **information gain**:

```
H(y)   = −p log₂ p − (1−p) log₂ (1−p)
IG     = H(parent) − ( |L|/N · H(L) + |R|/N · H(R) )
```

- **Candidate thresholds** are the unique feature values; if there are more than `max_thresholds`, a random subset is used. This keeps the search fast on high-cardinality features.
- A node becomes a **leaf** when any of these holds: fewer than `min_samples_split` samples, all labels equal (pure), `max_depth` reached, or no split improves information gain.
- Leaves store the **probability of class 1**; prediction thresholds it at 0.5.
- The tree records *why* branches stopped, which helps when tuning:

```
Termination by depth:   ...
Termination by samples: ...
Termination by purity:  ...
```

### 3. Random Forest (`random_forest.py`)

```
for each of n_estimators trees:
    1. draw a bootstrap sample (N rows, with replacement)
    2. grow a tree where every split considers only max_features random features
prediction = mean of tree probabilities ≥ 0.5
```

Two sources of randomness decorrelate the trees:

| Source | Effect |
|---|---|
| **Bootstrap sampling** | each tree sees a different version of the data, reducing variance |
| **Feature subsampling** | stops one strong feature from dominating every tree |
| **Random threshold sampling** | cheaper search and extra diversity |

---

## 💻 Usage Examples

**Run the experiments**

```bash
pip install -r requirements.txt
python train_decision_tree.py
python train_random_forest.py     # slower: grows 200 trees
```

**Use the classes directly**

```python
from decision_tree import DecisionTree
from random_forest import RandomForest

tree = DecisionTree(min_samples_split=400, max_depth=7, max_thresholds=200)
tree.fit(X_train, y_train)
print("Tree accuracy:", (tree.predict(X_test) == y_test).mean())
tree.print_termination_stats()

forest = RandomForest(n_estimators=200, min_samples_split=400,
                      max_depth=6, max_features=6, max_thresholds=200)
forest.fit(X_train, y_train)
print("Forest accuracy:", (forest.predict(X_test) == y_test).mean())
```

---

## 📈 Results & Hyperparameter Study

| Model | Test accuracy | Settings |
|---|---|---|
| Decision tree | ≈ 63.8% | `max_depth=7`, `min_samples_split=400` |
| Random forest | ≈ 64.1% | 200 trees, `max_depth=6`, `max_features=6`, `min_samples_split=400` |

### Decision tree: effect of regularisation

- **Very small `min_samples_split`** → large trees, high variance, overfitting
- **Very large `min_samples_split`** → underfitting and lower accuracy
- **Best region:** depths of 4–6 and `min_samples_split` between 200 and 600, giving stable test accuracy around 63–64%

### Random forest: effect of hyperparameters

- Accuracy improves with more trees up to about **200**, after which gains become marginal
- **Moderate depth and split size** work best; very shallow trees underfit, very restrictive splitting hurts
- **Too few features per split** weakens individual trees and lowers ensemble accuracy; some randomness helps, too much hurts

---

## 🔍 Key Insights

- The forest consistently beats the single tree, and the gain comes from **variance reduction** rather than a more powerful base learner
- Bootstrapping and feature subsampling make the trees diverse, which stabilises predictions and improves generalisation
- The accuracy ceiling is low (~64%) because the features are mostly identifiers and schedule data, which carry limited signal about delays
- Encoding choices matter: ordinal codes for high-cardinality categories are simple and effective with threshold-based trees

---

## 📌 Implementation Notes

- Entropy is computed from the mean of the 0/1 labels, so binary targets are assumed
- Fixed seed 10 controls the split, bootstrap draws and threshold sampling
- Progress is printed every 10 trees during forest training

---

## 🧾 Conclusion

Writing the tree and forest from scratch makes the effect of every design decision visible: stopping rules control bias and variance, information gain drives the splits, and randomness turns many weak, noisy trees into a more reliable ensemble.
