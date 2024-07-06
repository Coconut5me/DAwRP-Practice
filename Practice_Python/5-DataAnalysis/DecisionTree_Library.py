#%% - Import library
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import numpy as np

#%% - Function
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # For leaf nodes

class DecisionTreeClassifier:
    def __init__(self, criterion='gini', random_state=None, max_depth=None):
        self.criterion = criterion
        self.random_state = random_state
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0, prev_condition=""):
        np.random.seed(self.random_state)  # Set random seed
        n_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(n_samples_per_class)
        node = Node(value=predicted_class)

        # Print information about the current node
        print("Depth:", depth)
        print("Samples:", len(y))
        print("Class distribution:", n_samples_per_class)
        print("Predicted class:", predicted_class)

        # Calculate Gini impurity
        gini_impurity = self._gini(y)
        print("Gini impurity:", gini_impurity)

        # Stopping criteria
        if depth < self.max_depth:
            feature_indices = np.random.choice(self.n_features, self.n_features, replace=False)
            best_gain = 0
            for feature_index in feature_indices:
                thresholds = np.unique(X[:, feature_index])
                for threshold in thresholds:
                    gain, left_indices, right_indices = self._information_gain(X, y, feature_index, threshold)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature_index = feature_index
                        best_threshold = threshold
                        best_left_indices = left_indices
                        best_right_indices = right_indices

            if best_gain > 0:
                condition = prev_condition + f" {features[best_feature_index]} <= {best_threshold}"
                print("Splitting on feature", best_feature_index, "with threshold", best_threshold)
                print("Left node samples:", len(best_left_indices))
                print("Right node samples:", len(best_right_indices))
                print(condition)
                print()

                left = self._grow_tree(X[best_left_indices], y[best_left_indices], depth + 1)
                right = self._grow_tree(X[best_right_indices], y[best_right_indices], depth + 1)
                node = Node(feature_index=best_feature_index, threshold=best_threshold, left=left, right=right)

        return node

    def _information_gain(self, X, y, feature_index, threshold):
        parent_criterion = self._criterion(y)
        left_indices = np.where(X[:, feature_index] <= threshold)[0]
        right_indices = np.where(X[:, feature_index] > threshold)[0]

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0, None, None

        n = len(y)
        n_left = len(left_indices)
        n_right = len(right_indices)
        criterion_left = self._criterion(y[left_indices])
        criterion_right = self._criterion(y[right_indices])

        child_criterion = (n_left / n) * criterion_left + (n_right / n) * criterion_right
        information_gain = parent_criterion - child_criterion

        return information_gain, left_indices, right_indices

    def _criterion(self, y):
        if self.criterion == 'gini':
            return self._gini(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)

    def _gini(self, y):
        if len(y) == 0:
            return 0
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p ** 2)

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        p = np.bincount(y) / len(y)
        return -np.sum(p * np.log2(p + 1e-10))

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        X_numeric = self._transform_input(X)  # Convert input data to numeric format
        return np.array([self._predict(inputs) for inputs in X_numeric])

    def _transform_input(self, X):
        # Implement input transformation here (e.g., using LabelEncoder)
        # Ensure X is converted to a numeric format compatible with the thresholds used during training
        return X

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

#%% - Load data
df=pd.read_csv("data/Buys_computer.csv")
x = df.drop('Buys_computer', axis='columns')
y = df['Buys_computer']

#%% - Preprocessing data
from sklearn.preprocessing import LabelEncoder
x['Age_n'] = LabelEncoder().fit_transform(x['Age'])
x['Income_n'] = LabelEncoder().fit_transform(x['Income'])
x['Student_n'] = LabelEncoder().fit_transform(x['Student'])
x['Credit_rating_n'] = LabelEncoder().fit_transform(x['Credit_rating'])
x_n=x.drop(['Age','Income','Student','Credit_rating'], axis='columns')
y_n=LabelEncoder().fit_transform(y)

# Convert x_n and y_n to numpy arrays
x_n_array = x_n.values
y_n_array = y_n

# Ensure y_n is a 1-dimensional array
y_n_array = np.ravel(y_n_array)

#%% - Fit model
features = ['Age','Income','Student','Credit_rating']
model = DecisionTreeClassifier(criterion='gini', random_state=10, max_depth=3)
model.fit(x_n_array, y_n_array)

#%%
score = model.score(x_n_array, y_n)

#%% - Prediction
# Age <= 30, Income: Low, Student: yes, Credict:fair?
buy_computer = model.predict([[1,1,1,1]])
buy_computer