import numpy as np

# f1_score():
# input:
#       y_cv = Cross Validation data labels
#       y_predicted = predicted data labels
# output:
#       score = F1 Score
#       precision = Precision score
#       recall = Recall score
# Description: calculates the score, precision & recall of the training algorithm


def f1_score(y_cv, y_predicted):

    # true_positive = how many samples we classified "1" right (predicted = 1, actual = 1)
    # false_positive = how many samples we classified "1" wrong - "1" instead of "0" (predicted = 1, actual = 0)
    # false_negative = how many samples we classified "0" wrong - 0 instead of 1 (predicted = 0, actual = 1)
    # code for: true_positive = 11, false_positive = 10, false_negative = 01
    # assuming the vectors store only "1" or "0"
    measures_vector = y_predicted * 10 + y_cv

    true_positive = sum(measures_vector == 11)
    false_positive = sum(measures_vector == 10)
    false_negative = sum(measures_vector == 1)

    precision = true_positive / (
            true_positive + false_positive)  # percentage of true classification, out of all "1" classified
    recall = true_positive / (
            true_positive + false_negative)  # percentage of true classification, out of all actual "1"
    score = 2 * ((precision * recall) / (precision + recall))  # @@@ COMMENT @@@

    return score, precision, recall


class LinearSVM:

# Constructor

    def __init__(self, alpha_param=0.001, lambda_param=0.01, n_iterations=30000):
        self.alpha_param = alpha_param
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None

# training():
# input:
#       X = data set
#       y = data labels
# output:
#       None
# Description: finds weights and bias of a hyperplane with largest possible margins for classification between 2 classes

    def training(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)  # adjust y values to -1 or 1 for classification

        self.w = np.zeros(n_features)
        self.b = 0

        #         epsilon_param = 0.000001
        epsilon_param = 3e-6
        epsilon_flag = True
        current_iteration = 0
        J_previous = 9999

        while (current_iteration < self.n_iterations) and epsilon_flag:

            hingeLoss = 0

            for i, sample in enumerate(X):

                # @@@@@@@@@ COMMENT @@@@@@@@@
                hyperplaneFunc = y_[i] * (np.dot(sample, self.w) - self.b) >= 1
                hingeLoss += max(0, 1 - hyperplaneFunc)

                gradient_0_w = 2 * self.lambda_param * self.w  # dJ/dw = 2λw
                gradient_0_b = 0  # dJ/db = 0
                gradient_1_w = 2 * self.lambda_param * self.w - np.dot(sample, y_[i])  # dJ/dw = 2λw - yixi
                gradient_1_b = y_[i]  # dJ/db = yi

                # gradient descent
                if hyperplaneFunc:
                    self.w -= self.alpha_param * gradient_0_w  # w = w - α * dJ/dw
                    self.b -= self.alpha_param * gradient_0_b  # b = b - α * dJ/db
                else:
                    self.w -= self.alpha_param * gradient_1_w  # w = w - α * dJ/dw
                    self.b -= self.alpha_param * gradient_1_b  # b = b - α * dJ/db

            current_iteration += 1
            J_current = (hingeLoss / n_samples) + (self.lambda_param * (np.linalg.norm(self.w) ** 2))
            epsilon_flag = (abs(J_current - J_previous)) > epsilon_param
            J_previous = J_current

        return self.w, self.b, current_iteration

# predict():
# input:
#       X = data set
# output:
#       sign of y=w*x-b
# Description: calculates the value of the hyperplane function (y=w*x-b) and returns its sign

    def predict(self, X):
        prediction = np.dot(X, self.w) - self.b
        return np.sign(prediction)
