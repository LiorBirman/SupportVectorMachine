import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from SupportVectorMachine import SVM


def visualize_svm():
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    color = np.where(y >= 1, 'r', 'b')
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=color)

    color = np.where(y_cv >= 1, 'r', 'b')
    plt.scatter(X_cv[:, 0], X_cv[:, 1], marker='x', c=color)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, svm.w, svm.b, 0)
    x1_2 = get_hyperplane_value(x0_2, svm.w, svm.b, 0)

    x1_1_m = get_hyperplane_value(x0_1, svm.w, svm.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, svm.w, svm.b, -1)

    x1_1_p = get_hyperplane_value(x0_1, svm.w, svm.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, svm.w, svm.b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'y--')
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k')
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k')

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.show()


X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=2, random_state=40)
svm = SVM()
svm.training(X, y)

X_cv, y_cv = datasets.make_blobs(n_samples=30, n_features=2, centers=2, cluster_std=5, random_state=40)

y_predicted = svm.predict(X_cv)

print(svm.f1_score(y_cv, y_predicted))

visualize_svm()