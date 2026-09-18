import numpy as np

X = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
])
y = np.array([0, 0, 1, 1])

x = np.array([8.0, 4.0])
k = 3
prediction = np.bincount(y[np.argsort(np.linalg.norm(X - x, axis=1))[:k]]).argmax()

print(f"Predicted class for point {x} is: {prediction}")
