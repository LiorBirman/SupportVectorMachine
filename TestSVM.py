import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from SupportVectorMachine import SVM
import Plotting as p

# trainAndPredict() function receives a set of values that are used to make data ("." for training "x" crosses for predictions)
exampleNumber = 1
def trainPredictPlot(n_samples_1=150, n_features_1=2, centers_1=2, cluster_std_1=2, random_state_1=40,
                    n_samples_2=30, n_features_2=2, centers_2=2, cluster_std_2=5, random_state_2=40):

    # create data for training & predictions
    X, y = datasets.make_blobs(n_samples=n_samples_1, n_features=n_features_1, centers=centers_1, cluster_std=cluster_std_1, random_state=random_state_1)
    X_cv, y_cv = datasets.make_blobs(n_samples=n_samples_2, n_features=n_features_2, centers=centers_2, cluster_std=cluster_std_2, random_state=random_state_2)

    # create SVM object
    svm = SVM()

    # train classifications with X & y
    svm.training(X, y)

    # predict classifications with X_cv & y_cv
    y_predicted = svm.predict(X_cv)

    print("Example number ", exampleNumber, ":")
    print("F1_Score, Precision, Recall: ", svm.f1_score(y_cv, y_predicted))

    p.plotSVM(X, X_cv, y, y_cv, svm, exampleNumber)


trainPredictPlot()
exampleNumber += 1