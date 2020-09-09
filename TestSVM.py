import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from SupportVectorMachine import LinearSVM
from SupportVectorMachine import f1_score
import Plotting as p

# dataGenerator():
# input:
#       n_samples = number of samples to generate
#       cluster_std = standard deviation spread
#       n_features = number of features to generate
#       centers = number of classes
#       random_state = determines random number generation
# output:
#       X = Data Sets
#       y = Data Labels
# Description: generates data set & data labels based on input parameters


def dataGenerator(n_samples, cluster_std, n_features=2, centers=2, random_state=40):

    # create data for training & predictions
    X, y = datasets.make_blobs(n_samples=n_samples, n_features=n_features, centers=centers,
                               cluster_std=cluster_std, random_state=random_state)

    return X, y


# predictAndPlot():
# input:
#       X = data set
#       y = data labels
#       n_samples = number of samples to generate
#       cluster_std = standard deviation spread
#       fig_title = title of plotted figure
#       svm = SVM class object
# output:
#       None
# Description: generates data for prediction, predicts the classification and draws on screen


def predictAndPlot(X, y, n_samples, cluster_std, fig_title, w, b, iters):
    # prediction data generation
    X_test, y_test = dataGenerator(n_samples, cluster_std)

    # predict classifications with X_test & y_test data
    y_predicted = svm_object.predict(X_test)

    p.plotSVM(X, X_test, y, y_test, w, b, fig_title, y_predicted, iters)


# create SVM object
svm_object = LinearSVM()

# training data generation
samples = 150
clusterStd = 2
X, y = dataGenerator(samples, clusterStd)

# train classifications with X & y data
w, b, iterations = svm_object.training(X, y)

# cross validation
samples = 30
clusterStd = 2
figure_title = "Training & Classification"
predictAndPlot(X, y, samples, clusterStd, figure_title, w, b, iterations)

# test with noise
samples = 30
clusterStd = 3
figure_title = "Noisy Data Test"
predictAndPlot(X, y, samples, clusterStd, figure_title, w, b, iterations)

# test with bigger noise
samples = 30
clusterStd = 7
figure_title = "Bigger Noise Data Test"
predictAndPlot(X, y, samples, clusterStd, figure_title, w, b, iterations)
