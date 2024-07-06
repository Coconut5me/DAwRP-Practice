import numpy as np
class Node:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        info_gain=None,
        value=None,
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, mode="entropy"):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.mode = mode

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root

        if node.value is not None:
            print(f"{node.value}")
        else:
            print(f"X{node.feature_index} <= {node.threshold}")
            print(f"Info Gain: {node.info_gain}")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and depth <= self.max_depth:
            best_split = self._get_gest_split(X, y)
            if best_split["info_gain"] > 0:
                left = self._grow_tree(
                    X[best_split["left_indexes"]],
                    y[best_split["left_indexes"]],
                    depth + 1,
                )
                right = self._grow_tree(
                    X[best_split["right_indexes"]],
                    y[best_split["right_indexes"]],
                    depth + 1,
                )
                return Node(
                    best_split["feature_index"],
                    best_split["threshold"],
                    left,
                    right,
                    best_split["info_gain"],
                )

        leaf_value = self._most_common_label(y)
        return Node(value=leaf_value)

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _get_gest_split(self, X, y):
        best_split = {}
        max_info_gain = -float("inf")
        n_samples, n_features = np.shape(X)

        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            unique_values = np.unique(feature_values)
            for threshold in unique_values:
                left_indexes, right_indexes = self._split(feature_values, threshold)
                if len(left_indexes) > 0 or len(right_indexes) > 0:
                    info_gain = self._information_gain(y, left_indexes, right_indexes)
                    if info_gain > max_info_gain:
                        best_split = {
                            "feature_index": feature_index,
                            "threshold": threshold,
                            "left_indexes": left_indexes,
                            "right_indexes": right_indexes,
                            "info_gain": info_gain,
                        }

        return best_split

    def _information_gain(self, y, left_indexes, right_indexes):
        left_weights = len(left_indexes) / len(y)
        right_weights = len(right_indexes) / len(y)

        parent_loss = self._entropy(y) if self.mode == "entropy" else self._gini(y)
        left_loss = (
            self._entropy(y[left_indexes])
            if self.mode == "entropy"
            else self._gini(y[left_indexes])
        )
        right_loss = (
            self._entropy(y[right_indexes])
            if self.mode == "entropy"
            else self._gini(y[right_indexes])
        )

        return parent_loss - (left_weights * left_loss + right_weights * right_loss)

    @staticmethod
    def _entropy(y):
        class_labels = np.unique(y)
        n_samples = len(y)
        entropy = 0
        for class_label in class_labels:
            p = len(y[y == class_label]) / n_samples
            entropy += -p * np.log2(p)
        return entropy

    @staticmethod
    def _gini(y):
        class_labels = np.unique(y)
        n_samples = len(y)
        gini = 0
        for class_label in class_labels:
            p = len(y[y == class_label]) / n_samples
            gini += p**2
        return 1 - gini

    @staticmethod
    def _split(feature_values, threshold):
        left_indexes = np.argwhere(feature_values <= threshold).flatten()
        right_indexes = np.argwhere(feature_values > threshold).flatten()
        return left_indexes, right_indexes

    @staticmethod
    def _most_common_label(y):
        return np.bincount(y).argmax()
