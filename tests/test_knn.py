import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from knn_model import KNN, StandardScaler


@pytest.fixture(scope="module")
def iris_split():
    iris = load_iris()
    return train_test_split(iris.data, iris.target, test_size=0.3,
                            stratify=iris.target, random_state=42)


@pytest.mark.parametrize("k", [1, 3, 5, 7, 9, 15])
def test_matches_sklearn_for_odd_k(iris_split, k):
    X_tr, X_te, y_tr, _ = iris_split
    ours = KNN(k).fit(X_tr, y_tr).predict(X_te)
    theirs = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr).predict(X_te)
    np.testing.assert_array_equal(ours, theirs)


def test_tie_goes_to_closest_class():
    # Two neighbours, one per class: the closer one (class 1 at x=1) must win.
    X = np.array([[1.0], [3.0]])
    y = np.array([1, 0])
    assert KNN(k=2).fit(X, y).predict([[0.0]])[0] == 1


def test_string_labels():
    X = np.array([[0.0], [0.1], [5.0]])
    y = np.array(["cat", "cat", "dog"])
    assert KNN(k=3).fit(X, y).predict([[4.9]])[0] == "cat"
    assert KNN(k=1).fit(X, y).predict([[4.9]])[0] == "dog"


def test_scaler_mean_zero_std_one():
    X = np.array([[40, 500], [60, 350], [90, 300]], dtype=float)
    Z = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0), 1)


def test_scaler_constant_column():
    Z = StandardScaler().fit_transform([[1, 7], [2, 7], [3, 7]])
    assert np.all(np.isfinite(Z)) and np.all(Z[:, 1] == 0)


def test_rejects_bad_k():
    with pytest.raises(ValueError):
        KNN(k=0)
    with pytest.raises(ValueError):
        KNN(k=5).fit([[0], [1]], [0, 1])


def test_restaurant_example_scaling_changes_answer():
    import knnv2
    raw = KNN(knnv2.K).fit(knnv2.X, knnv2.y).predict(knnv2.restaurant)[0]
    scaler = StandardScaler().fit(knnv2.X)
    scaled = KNN(knnv2.K).fit(scaler.transform(knnv2.X), knnv2.y) \
        .predict(scaler.transform(knnv2.restaurant))[0]
    assert (raw, scaled) == (0, 1)
