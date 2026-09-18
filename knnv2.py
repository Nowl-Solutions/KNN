import numpy as np
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

restaurant = np.array([10, 380])
distance = np.sqrt(np.sum((X - restaurant) ** 2, axis=1))
K = 5
nearest = np.argsort(distance)[:K]
# Majority Vote
prediction = np.mean(y[nearest])
print(prediction)
if prediction >= 0.5:
    print("Expensive")
else:
    print("Cheap")
