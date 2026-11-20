"""K-Nearest Neighbors written with NumPy only.

    model = KNN(k=5).fit(X_train, y_train)
    model.predict(X_test)
"""
import numpy as np


class StandardScaler:
    """Rescale every feature to mean 0, standard deviation 1.

    KNN measures distance, so a feature measured in hundreds (e.g. distance in metres)
    would otherwise drown out a feature measured in tens (e.g. price in baht).
    """

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0          # a constant column stays as-is
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class KNN:
    def __init__(self, k=5):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        if len(self.X_) < self.k:
            raise ValueError(f"k={self.k} but only {len(self.X_)} training points")
        return self

    def kneighbors(self, X):
        """Indices and distances of the k nearest training points for each row of X."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        # (n_test, n_train) Euclidean distances in one shot
        d = np.sqrt(((X[:, None, :] - self.X_[None, :, :]) ** 2).sum(axis=2))
        idx = np.argsort(d, axis=1, kind="stable")[:, :self.k]
        return idx, np.take_along_axis(d, idx, axis=1)

    def predict(self, X):
        idx, _ = self.kneighbors(X)
        return np.array([self._vote(self.y_[row]) for row in idx])

    @staticmethod
    def _vote(labels):
        """Majority vote over neighbour labels (closest first).

        On a tie, the tied class whose member is closest wins, because the
        neighbours are ordered by distance and we scan them in that order.
        """
        classes, counts = np.unique(labels, return_counts=True)
        tied = set(classes[counts == counts.max()])
        return next(label for label in labels if label in tied)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
