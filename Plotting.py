import numpy as np
import matplotlib.pyplot as plt
import SupportVectorMachine as SVM

# getHyperplane():
# input:
#       X = data set
#       w = features weights
#       b = hyperplane bias
#       offset = margin offset
# output:
#       projection of "x2"
# Description: calculates and returns the projection of "x2"

def getHyperplane(X, w, b, offset):
    return (-w[0] * X + b + offset) / w[1]

# calculateHyperplaneAndMargins():
# input:
#       X = data set
#       svm = LinearSVM object
# output:
#       (x1_min, x2_hyperplane_min) & (x1_max, x2_hyperplane_max)= hyperplane edge points
#       (x1_min, x2_bottom_min) & (x1_max, x2_bottom_max) = bottom margin edge points
#       (x1_min, x2_top_min) & (x1_max, x2_top_max) = top margin edge points
# Description: calculates and returns hyperplane & margins

def calculateHyperplaneAndMargins(X, svm):
    # calculates the left edge points of hyperplane & margins
    x1_min = np.amin(X[:, 0])
    x2_hyperplane_min = getHyperplane(x1_min, svm.w, svm.b, 0)
    x2_bottom_min = getHyperplane(x1_min, svm.w, svm.b, -1)
    x2_top_min = getHyperplane(x1_min, svm.w, svm.b, 1)

    # calculates the right edge points of hyperplane & margins
    x1_max = np.amax(X[:, 0])
    x2_hyperplane_max = getHyperplane(x1_max, svm.w, svm.b, 0)
    x2_bottom_max = getHyperplane(x1_max, svm.w, svm.b, -1)
    x2_top_max = getHyperplane(x1_max, svm.w, svm.b, 1)

    return x1_min, x1_max, x2_hyperplane_min, x2_hyperplane_max, x2_bottom_min, x2_bottom_max, x2_top_min, x2_top_max


def plotLines(ax, x0_1, x0_2, x1_1, x1_2, x_1_1_bottom, x_1_2_bottom, x_1_1_top, x_1_2_top):
    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'y--')
    ax.plot([x0_1, x0_2], [x_1_1_bottom, x_1_2_bottom], 'k')
    ax.plot([x0_1, x0_2], [x_1_1_top, x_1_2_top], 'k')


def plotData(X, y, marker):
    color = np.where(y >= 1, 'r', 'b')
    plt.scatter(X[:, 0], X[:, 1], marker=marker, c=color)


def plotSVM(X, X_generated, y, y_generated, svm, fig_title, y_predicted):
    fig = plt.figure()
    fig.suptitle(fig_title, fontsize=40)

    x0_1, x0_2, x1_1, x1_2, x_1_1_bottom, x_1_2_bottom, x_1_1_top, x_1_2_top = calculateHyperplaneAndMargins(X, svm)

    ax = fig.add_subplot(2, 2, 1)
    ax.set_title("Training", fontsize=25)

    # plot training, hyperplane & margins
    marker = 'o'
    plotData(X, y, marker)
    plotLines(ax, x0_1, x0_2, x1_1, x1_2, x_1_1_bottom, x_1_2_bottom, x_1_1_top, x_1_2_top)

    ax = fig.add_subplot(2, 2, 2)
    ax.set_title("Prediction", fontsize=25)

    # plot prediction, hyperplane & margins
    marker = 'x'
    plotData(X_generated, y_generated, marker)
    plotLines(ax, x0_1, x0_2, x1_1, x1_2, x_1_1_bottom, x_1_2_bottom, x_1_1_top, x_1_2_top)

    ax = fig.add_subplot(2, 2, 4)
    score, precision, recall = SVM.f1_score(y_generated, y_predicted)
    score_text = "F1_Score: {}".format(score)
    recall_text = "Recall: {}".format(recall)
    precision_text = "Precision: {}".format(precision)
    plt.text(0.018, 0.893, score_text, fontsize=20)
    plt.text(0.018, 0.693, recall_text, fontsize=20)
    plt.text(0.018, 0.593, precision_text, fontsize=20)

    plt.show()
