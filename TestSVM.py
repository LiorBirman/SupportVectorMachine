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


# create SVM object
alpha_param = 0.0001
lambda_param = 0.01
n_iterations = 15000
svm_object = LinearSVM(alpha_param=alpha_param, lambda_param=lambda_param, n_iterations=n_iterations)

# data generation
samples = 500
clusterStd = 2
whole_data, whole_labels = dataGenerator(samples, clusterStd)

# 70% for training, 30% for cross validation
training_portion = int(samples * 0.7)
prediction_portion = int(samples * 0.3)

# training data
X = whole_data[0:training_portion, :]
y = whole_labels[0:training_portion]

# cross validation data
X_cv = whole_data[training_portion::, :]
y_cv = whole_labels[training_portion::]

# train classifications with training data
w, b, iterations = svm_object.training(X, y)

# predict classifications of cross validation data
y_predicted = svm_object.predict(X_cv)

# show on screen
figure_title = "Training & Classification"
p.plotSVM(X, X_cv, y, y_cv, w, b, figure_title, y_predicted, iterations, lambda_param, alpha_param, samples, training_portion, prediction_portion)


# new prediction data generation
samples = 166
clusterStd = 5
X_test, y_test = dataGenerator(samples, clusterStd)

# predict classifications of cross validation data
y_predicted = svm_object.predict(X_test)

# show on screen
figure_title = "Original Training & Noisy Data Classification"
p.plotSVM(X, X_test, y, y_test, w, b, figure_title, y_predicted, iterations, lambda_param, alpha_param, samples, training_portion, prediction_portion)
