import numpy as np
import matplotlib.pyplot as plt

def getHyperplane(X, w, b, offset):
    return (-w[0] * X + b + offset) / w[1]
def getLines(X, svm):
    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    # Hyperplane
    x1_1 = getHyperplane(x0_1, svm.w, svm.b, 0)
    x1_2 = getHyperplane(x0_2, svm.w, svm.b, 0)

    # Bottom Margin
    x_1_1_bottom = getHyperplane(x0_1, svm.w, svm.b, -1)
    x_1_2_bottom = getHyperplane(x0_2, svm.w, svm.b, -1)

    # Top Margin
    x_1_1_top = getHyperplane(x0_1, svm.w, svm.b, 1)
    x_1_2_top = getHyperplane(x0_2, svm.w, svm.b, 1)

    return x0_1, x0_2, x1_1, x1_2, x_1_1_bottom, x_1_2_bottom, x_1_1_top, x_1_2_top
def plotLines(ax, x0_1, x0_2, x1_1, x1_2, x_1_1_bottom, x_1_2_bottom, x_1_1_top, x_1_2_top):
    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'y--')
    ax.plot([x0_1, x0_2], [x_1_1_bottom, x_1_2_bottom], 'k')
    ax.plot([x0_1, x0_2], [x_1_1_top, x_1_2_top], 'k')
def plotData(X, y):
    color = np.where(y >= 1, 'r', 'b')
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=color)

def plotSVM(X, X_cv, y, y_cv, svm, exampleNumber):
    example = "Example Number {}:".format(exampleNumber)

    fig = plt.figure()
    fig.suptitle(example, fontsize=40)

    x0_1, x0_2, x1_1, x1_2, x_1_1_bottom, x_1_2_bottom, x_1_1_top, x_1_2_top = getLines(X, svm)

    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Training", fontsize=25)

    # plot training, hyperplane & margins
    plotData(X, y)
    plotLines(ax, x0_1, x0_2, x1_1, x1_2, x_1_1_bottom, x_1_2_bottom, x_1_1_top, x_1_2_top)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Predictions", fontsize=25)

    # plot training, predictions, hyperplane & margins
    plotData(X, y)
    plotData(X_cv, y_cv)
    plotLines(ax, x0_1, x0_2, x1_1, x1_2, x_1_1_bottom, x_1_2_bottom, x_1_1_top, x_1_2_top)

    plt.show()