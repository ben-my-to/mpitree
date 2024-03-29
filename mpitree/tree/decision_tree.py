"""A module that defines the (parallel) decision tree classifier classes.

Author: Jason Duong Date: 01/11/24
"""

from __future__ import annotations

from typing import override

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._base import BranchType, Node


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """A decision tree classifier class.

    Parameters
    ----------
    max_depth : int, optional
        A hyperparameter that upper bounds the number of splits per level.
        (the default is `None`, which implies the decision tree can be at
        any depth).

    min_samples_split : int, optional
        A hyperparameter that lower bounds the number of samples required
        to split. (the default is 2, which implies decision tree may
        contain singleton nodes).
    """

    def __init__(self, *, max_depth: int = None, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _compute_entropy(self, x):
        """Measure the impurity on an array.

        Parameters
        ----------
        x : array-like
            Feature column or target array of shape (n_samples,).

        Returns
        -------
        float
        """
        _, n_unique_classes = np.unique(x, return_counts=True)
        proba = n_unique_classes / len(x)
        return -np.sum(proba * np.log2(proba))

    def _compute_information_gain(self, X, y, feature_idx):
        """Return the difference of entropies before and after splitting.

        Parameters
        ----------
        X : array-like
            Feature array with shape (n_samples, n_features) of
            numerical values.

        y : array-like
            Target array with shape (n_samples,) of categorical values.

        feature_idx : int
            The feature index value.

        Returns
        -------
        tuple
            The maximum information gain and optimal threshold value.
        """
        possible_thresholds = np.unique(X[:, feature_idx])

        costs = []
        for threshold in possible_thresholds:
            region = X[:, feature_idx] <= threshold
            levels = X[region], X[~region]
            weights = np.array([len(level) / len(X) for level in levels])
            impurity = np.array(
                [
                    self._compute_entropy(y[region]),
                    self._compute_entropy(y[~region]),
                ]
            )
            costs.append(np.dot(weights, impurity))

        return (
            self._compute_entropy(y) - min(costs),
            possible_thresholds[np.argmin(costs)],
        )

    def _make_tree(self, X, y, *, parent=None, depth=0):
        """Recursively constructs a decision tree classifier.

        Parameters
        ----------
        X : array-like
            Feature array with shape (n_samples, n_features) of
            numerical values.

        y : array-like
            Target array with shape (n_samples,) of categorical values.

        parent : DecisionTreeClassifier, default=None
            The precedent node.

        depth : int, default=0
            The number of edges from the root node.

        Returns
        -------
        Node
        """
        n_classes = len(np.unique(y))
        n_samples = len(X)

        if (
            n_classes == 1
            or np.all(X == X[0])
            or (self.max_depth is not None and self.max_depth == depth)
            or n_samples < self.min_samples_split
        ):
            return Node(
                value=np.bincount(y).argmax(),
                count=np.bincount(y, minlength=len(self.classes_)),
                parent=parent,
            )

        best_level_gains = []
        best_level_thresholds = []

        for feature_idx in range(self.n_features_):
            gain_pred, threshold_pred = self._compute_information_gain(
                X, y, feature_idx
            )
            best_level_gains.append(gain_pred)
            best_level_thresholds.append(threshold_pred)

        split_feature_idx = np.argmax(best_level_gains)
        split_threshold = best_level_thresholds[split_feature_idx]

        split_node = Node(
            value=split_feature_idx,
            threshold=split_threshold,
            count=np.bincount(y, minlength=len(self.classes_)),
            parent=parent,
        )

        region = X[:, split_node.value] <= split_node.threshold

        split_node.left = self._make_tree(
            X[region],
            y[region],
            parent=split_node,
            depth=depth + 1,
        )

        split_node.right = self._make_tree(
            X[~region],
            y[~region],
            parent=split_node,
            depth=depth + 1,
        )

        return split_node

    def fit(self, X, y):
        """Train a decision tree classifier.

        Parameters
        ----------
        X : array-like
            Feature array with shape (n_samples, n_features) of
            numerical values.

        y : array-like
            Target array with shape (n_samples,) of categorical values.

        Returns
        -------
        DecisionTreeClassifier
        """
        X, y = check_X_y(X, y, dtype=object)

        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)

        self.tree_ = self._make_tree(X, y)
        return self

    def predict_proba(self, X):
        """Return the array of occurrences of each class.

        Parameters
        ----------
        X : array-like
            Test feature array with shape (n_samples, n_features) of
            numerical values.

        Returns
        -------
        ndarray
        """
        check_is_fitted(self)
        X = check_array(X, dtype=object)

        def walk(x, node=self.tree_) -> Node:
            """Traverse a decision tree from the root to some leaf node.

            Nodes are queried based on the respective feature value from
            `x` at each level until some leaf node has reached.

            Parameters
            ----------
            x : array-like
                Test instance array of shape (1, n_features).

            Returns
            -------
            Node
            """
            if node.is_leaf:
                return node.count
            return walk(x, node.left if x[node.value] <= node.threshold else node.right)

        return np.apply_along_axis(walk, axis=1, arr=X)

    def predict(self, X):
        """Return the array of predicted leaf nodes.

        The function returns the class with highest probability for each
        test sample.

        Parameters
        ----------
        X : array-like
            Test feature array with shape (n_samples, n_features) of
            numerical values.

        Returns
        -------
        ndarray
        """
        check_is_fitted(self)
        X = check_array(X, dtype=object)

        return np.argmax(self.predict_proba(X), axis=1)

    def export_text(self, *, feature_names=None, class_names=None, precision=2):
        """Return a text-based visualization of a decision tree classifier.

        Parameters
        ----------
        feature_names : array-like
            Array of shape (1, n_features).

        class_names : array-like
            Array of shape (1, n_classes).

        precision : int
            The number of digits right of the decimal point.

        Returns
        -------
        str
        """
        check_is_fitted(self)

        def content(node, value):
            data = " ".join((node._btype.value, value))

            if not node.depth:
                return data
            sign = "<=" if node is node.parent.left else ">"
            return f"{data} [{sign} {node.parent.threshold:.{precision}f}]"

        def apply_prefixer(node, result=None, prefix=""):
            if result is None:
                result = []

            # branch = node._btype.value

            if node.is_leaf:
                value = (
                    class_names[node.value]
                    if class_names is not None
                    else f"class: {node.value}"
                )
            else:
                value = (
                    feature_names[node.value]
                    if feature_names is not None
                    else f"feature_{node.value}"
                )

            result.append(prefix + content(node, value))

            for child in sorted(node.children):
                if node._btype is BranchType.LEAF_LIKE:
                    apply_prefixer(child, result, prefix + "   ")
                else:
                    apply_prefixer(child, result, prefix + "│  ")

            return "\n".join(result)

        return apply_prefixer(self.tree_)


class ParallelDecisionTreeClassifier(DecisionTreeClassifier):
    """A parallel decision classifier class."""

    from mpi4py import MPI

    WORLD_COMM = MPI.COMM_WORLD
    WORLD_RANK = WORLD_COMM.Get_rank()
    WORLD_SIZE = WORLD_COMM.Get_size()

    def _get_cyclic_dist(
        self, comm: MPI.Intracomm = None, *, n_blocks: int = 1
    ) -> MPI.Intracomm:
        """Schedules processes in a round-robin fashion.

        Parameters
        ----------
        comm : MPI.Intracomm, default=None
            The current communicator.

        n_blocks : int, default=1
            The number of distributions.

        Returns
        -------
        MPI.Intracomm
        """
        rank = comm.Get_rank()
        key, color = divmod(rank, n_blocks)
        return comm.Split(color, key)

    def fit(self, X, y):
        """Train a decision tree classifier.

        Parameters
        ----------
        X : array-like
            Feature array with shape (n_samples, n_features) of
            numerical values.

        y : array-like
            Target array with shape (n_samples,) of categorical values.

        Returns
        -------
        DecisionTreeClassifier
        """
        X, y = check_X_y(X, y, dtype=object)

        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)

        self.tree_ = self._make_tree(X, y, comm=self.WORLD_COMM)
        return self

    @override
    def _make_tree(self, X, y, *, parent=None, depth=0, comm=WORLD_COMM):
        """Recursively constructs a parallel decision tree classifier.

        Parameters
        ----------
        X : array-like
            Feature array with shape (n_samples, n_features) of
            numerical values.

        y : array-like
            Target array with shape (n_samples,) of categorical values.

        parent : DecisionTreeClassifier, default=None
            The precedent node.

        depth : int, default=0
            The number of edges from the root node.

        comm : MPI.Intracomm, default=WORLD_COMM
            The current communicator.

        Returns
        -------
        Node
        """
        n_classes = len(np.unique(y))
        n_samples = len(X)

        rank = comm.Get_rank()
        size = comm.Get_size()

        if (
            n_classes == 1
            or np.all(X == X[0])
            or (self.max_depth is not None and self.max_depth == depth)
            or n_samples < self.min_samples_split
        ):
            return Node(
                value=np.bincount(y).argmax(),
                count=np.bincount(y, minlength=len(self.classes_)),
                parent=parent,
            )

        best_level_gains = []
        best_level_thresholds = []

        for feature_idx in range(self.n_features_):
            gain_pred, threshold_pred = self._compute_information_gain(
                X, y, feature_idx
            )
            best_level_gains.append(gain_pred)
            best_level_thresholds.append(threshold_pred)

        split_feature_idx = np.argmax(best_level_gains)
        split_threshold = best_level_thresholds[split_feature_idx]

        split_node = Node(
            value=split_feature_idx,
            threshold=split_threshold,
            count=np.bincount(y, minlength=len(self.classes_)),
            parent=parent,
        )

        region = X[:, split_node.value] <= split_node.threshold

        if size == 1:
            split_node.left = self._make_tree(
                X[region],
                y[region],
                parent=split_node,
                depth=depth + 1,
                comm=comm,
            )

            split_node.right = self._make_tree(
                X[~region],
                y[~region],
                parent=split_node,
                depth=depth + 1,
                comm=comm,
            )
        else:
            group = self._get_cyclic_dist(comm, n_blocks=2)

            if rank % 2 == 0:
                X, y = X[region], y[region]
                level = "<="
            else:
                X, y = X[~region], y[~region]
                level = ">"

            levels = comm.allgather(
                {
                    level: self._make_tree(
                        X,
                        y,
                        parent=split_node,
                        depth=depth + 1,
                        comm=group,
                    )
                }
            )

            for level in levels:
                for sign, subtree in level.items():
                    if sign == "<=":
                        split_node.left = subtree
                        split_node.left.parent = split_node
                    elif sign == ">":
                        split_node.right = subtree
                        split_node.right.parent = split_node

            group.Free()

        return split_node
