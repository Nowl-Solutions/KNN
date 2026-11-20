"""Cheap or expensive restaurant from two features.

Feature 1 ranges 40-90 and feature 2 ranges 300-500, so without scaling the
distance is decided almost entirely by feature 2.
"""
import numpy as np

from knn_model import KNN, StandardScaler 

X = np.array([
    [40, 500],
    [45, 450],
    [50, 500],
    [50, 400],
    [55, 450],
    [60, 500],
    [60, 350],
    [65, 400],
    [70, 350],
    [75, 400],
    [80, 350],
    [90, 300]
])

y = np.array([
    0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1
])
LABELS = {0: "Cheap", 1: "Expensive"}

# The old test point [10, 380] was outside the training range of feature 1 (40-90).
restaurant = np.array([[70, 450]])
K = 5

raw = KNN(k=K).fit(X, y)
idx, _ = raw.kneighbors(restaurant)
print("Without scaling:", LABELS[raw.predict(restaurant)[0]],
      "| neighbours", X[idx[0]].tolist())

scaler = StandardScaler().fit(X)
scaled = KNN(k=K).fit(scaler.transform(X), y)
idx, _ = scaled.kneighbors(scaler.transform(restaurant))
print("With scaling:   ", LABELS[scaled.predict(scaler.transform(restaurant))[0]],
      "| neighbours", X[idx[0]].tolist())
