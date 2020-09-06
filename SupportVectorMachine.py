import numpy as np

class SVM:

    def __init__(self, alpha_param=0.001, lambda_param=0.01, n_iterations=1000):
        self.alpha_param = alpha_param
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_features = X.shape[1]

        y_ = np.where(y <= 0, -1, 1)  # adjust y values to -1 or 1 for classification

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iterations):
            for i, sample in enumerate(X):
                gradient_0_w = 2 * self.lambda_param * self.w                     # dJ/dw = 2λw
                gradient_0_b = 0                                                  # dJ/db = 0
                gradient_1_w = 2 * self.lambda_param * self.w - y_[i] * X[i]      # dJ/dw = 2λw - yixi
                gradient_1_b = y_[i]                                              # dJ/db = yi
                hyperplaneFunc = y_[i] * (np.dot(sample, self.w) - self.b) >= 1   # J = λw^2 + sum(max(0, 1-yi(wx-b))

                if hyperplaneFunc:
                    self.w -= self.alpha_param * (2 * self.lambda_param * self.w)  # w -= dJ/dw
                    self.b -= 0  # b -= dJ/db
                else:
                    self.w -= self.alpha_param * (2 * self.lambda_param * self.w - np.dot(sample, y_[i]))  # w -= dJ/dw
                    self.b -= self.alpha_param * y_[i]  # b -= dJ/db

    def predict(self, X):
        prediction = np.dot(X, self.w) - self.b
        return np.sign(prediction)
