import numpy as np
import matplotlib.pyplot as plt
import SupportVectorMachine as SVM
from matplotlib import gridspec

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
#       w = features weights
#       b = hyperplane bias
# output:
#       (x1_min, x2_hyperplane_min) & (x1_max, x2_hyperplane_max)= hyperplane edge points
#       (x1_min, x2_bottom_min) & (x1_max, x2_bottom_max) = bottom margin edge points
#       (x1_min, x2_top_min) & (x1_max, x2_top_max) = top margin edge points
# Description: calculates and returns hyperplane & margins

def calculateHyperplaneAndMargins(X, w, b):
    # calculates the left edge points of hyperplane & margins
    x1_min = np.amin(X[:, 0])
    x2_hyperplane_min = getHyperplane(x1_min, w, b, 0)
    x2_bottom_min = getHyperplane(x1_min, w, b, -1)
    x2_top_min = getHyperplane(x1_min, w, b, 1)

    # calculates the right edge points of hyperplane & margins
    x1_max = np.amax(X[:, 0])
    x2_hyperplane_max = getHyperplane(x1_max, w, b, 0)
    x2_bottom_max = getHyperplane(x1_max, w, b, -1)
    x2_top_max = getHyperplane(x1_max, w, b, 1)

    return x1_min, x1_max, x2_hyperplane_min, x2_hyperplane_max, x2_bottom_min, x2_bottom_max, x2_top_min, x2_top_max


# plotLines():
# input:
#       (x1_min, x2_hyperplane_min) & (x1_max, x2_hyperplane_max)= hyperplane edge points
#       (x1_min, x2_bottom_min) & (x1_max, x2_bottom_max) = bottom margin edge points
#       (x1_min, x2_top_min) & (x1_max, x2_top_max) = top margin edge points
# output:
#       None
# Description: draws hyperplane & margins on window

def plotLines(ax, x1_min, x1_max, x2_hyperplane_min, x2_hyperplane_max, x2_bottom_min, x2_bottom_max, x2_top_min, x2_top_max):
    ax.plot([x1_min, x1_max], [x2_hyperplane_min, x2_hyperplane_max], 'y--')
    ax.plot([x1_min, x1_max], [x2_bottom_min, x2_bottom_max], 'k')
    ax.plot([x1_min, x1_max], [x2_top_min, x2_top_max], 'k')


# plotData():
# input:
#       X = data set
#       y = data labels
#       marker = data shape ("x" or "o")
# output:
#       None
# Description: draws data on window

def plotData(X, y, marker):
    color = np.where(y >= 1, 'r', 'b')
    plt.scatter(X[:, 0], X[:, 1], marker=marker, c=color)


# plotSVM():
# input:
#       X = training data set
#       X_test = prediction data set
#       y = training data labels
#       y_test = prediction data labels
#       w = features weights
#       b = hyperplane bias
#       fig_title = title of plotted figure
#       y_predicted = predicted data labels
# output:
#       window with drawn data
# Description: draws data input on window in different subplots

def plotSVM(X, X_cv, y, y_cv, w, b, fig_title, y_predicted, iters, lambda_param, alpha_param, samples):
    fig = plt.figure()
    fig.suptitle(fig_title, fontsize=40)

    x1_min, x1_max, x2_hyperplane_min, x2_hyperplane_max, x2_bottom_min, x2_bottom_max, x2_top_min, x2_top_max = calculateHyperplaneAndMargins(X, w, b)

    ax = fig.add_subplot(221)
    ax.set_title("Training", fontsize=25)

    ax.set_ylim([x1_min - 10, x1_max])

    # plot training, hyperplane & margins
    marker = 'o'
    plotData(X, y, marker)
    plotLines(ax, x1_min, x1_max, x2_hyperplane_min, x2_hyperplane_max, x2_bottom_min, x2_bottom_max, x2_top_min, x2_top_max)

    ax = fig.add_subplot(222)
    ax.set_title("Cross Validation", fontsize=25)
    ax.set_ylim([x1_min - 10, x1_max])

    # plot prediction, hyperplane & margins
    marker = 'x'
    plotData(X_cv, y_cv, marker)
    plotLines(ax, x1_min, x1_max, x2_hyperplane_min, x2_hyperplane_max, x2_bottom_min, x2_bottom_max, x2_top_min, x2_top_max)

    ax = fig.add_subplot(223)
    plt.text(0.018, 0.893, "Number Of Samples: {}".format(samples * 0.7), fontsize=15)
    plt.text(0.018, 0.793, "Training Iterations: {}".format(iters), fontsize=15)
    plt.text(0.018, 0.593, "Alpha Parameter: {}".format(alpha_param), fontsize=15)
    plt.text(0.018, 0.493, "Lambda Parameter: {}".format(lambda_param), fontsize=15)

    ax = fig.add_subplot(224)
    score, precision, recall = SVM.f1_score(y_cv, y_predicted)
    plt.text(0.018, 0.893, "Number Of Samples: {}".format(int(samples * 0.3)), fontsize=15)
    plt.text(0.018, 0.693, "F1_Score: {}".format(score), fontsize=15)
    plt.text(0.018, 0.593, "Recall: {}".format(recall), fontsize=15)
    plt.text(0.018, 0.493, "Precision: {}".format(precision), fontsize=15)


    plt.get_current_fig_manager().window.showMaximized()
    plt.show()
