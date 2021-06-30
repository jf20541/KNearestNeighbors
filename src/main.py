from collections import Counter
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


class KNearestNeighbors:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.x_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        """Euclidean Distance is a distance between n-points in Euclidean Space"""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def minkowski_distance_np(self, x1, x2, p):
        """Vector space generalized as both the Euclidean & Manhattan distance"""
        return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)

    def manhattan_distance(self, x1, x2):
        """Calculate the distance between vectors using the sum of their absolute difference"""
        return np.sum(np.abs(x1 - x2))

    def hamming_distance(self, x1, x2):
        """Calculate distance from comparing two binary data"""
        return np.sum(np.abs(x1 - x2)) / len(x1)

    def predict(self, x):
        """calling the predict label method to compute all samples as list comprehension
        Returns:
            [array]: predicted values
        """
        y_pred = [self.predict_label(x) for x in x]
        return np.array(y_pred)

    def predict_label(self, x):
        """Compute distance function on x_train, sort by distance,
            and extract the labels of KNN samples
        Args:
            x [float]: collect all x values using list comprehension
        Returns:
            [int]: return the majority vote class label
        """
        distances = [self.euclidean_distance(x, x_train) for x_train in self.x_train]
        y_indices = np.argsort(distances)[: self.k]
        k_closest_classes = [self.y_train[i] for i in y_indices]
        majority_vote = Counter(k_closest_classes).most_common(1)
        return majority_vote[0][0]

    def metric(self, y_true, y_pred):
        """
        Args:
            y_true [float]: y_true values
            y_pred [float]: prediction values
        Returns:
            [float]: accuracy metric for KNN performance
        """
        accuracy = (np.sum(y_true == y_pred) / len(y_true)) * 100
        return accuracy


if __name__ == "__main__":
    iris = datasets.load_iris()
    # normalize features
    features = normalize(iris.data)
    targets = iris.target
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2
    )
    # plot the graph of K-values [1,9] and its corresponding error rate
    error = []
    for i in range(1, 10):
        clf = KNearestNeighbors(k=i)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        error.append(pred != y_test)
    plt.plot(
        range(1, 10),
        error,
        color="red",
        linestyle="dashed",
        marker="o",
        markerfacecolor="blue",
        markersize=10,
    )
    plt.title("Error Rate K-Value")
    plt.xlabel("K-Value")
    plt.ylabel("Mean Error")
    plt.savefig("../plots/ErrorRateKValue.png")
    plt.show()
    print(f"K-Nearest Neighbor Accuracy: {clf.metric(y_test, pred):0.2f}%")