# Instructions:

# 1. Run: "TestSVM.py".
#    Results:
#               - Data Sets & Data Labels will be generated.
#               - SVM training will occur.
#               - Cross Validation data will be generated.
#               - Cross Validation data prediction will occur.
#               - The training and Cross Validation results will be shown on a window.

# 2. Close the window to proceed to the next test.
#    Results:
#               - Noisy data will be generated.
#               - Noisy data prediction will occur.
#               - The original training and the new data results will be shown on a window.

# 3.Change parameters in "main.py" to affect "TestSVM.py run"
#   alpha_param = learning rate.
#   lambda_param = regularization.
#   n_iterations = maximum iterations of training.
#   training_cv_samples = amount of data samples for training & cross validation (70%, 30%).
#   training_cv_clusterSTD = standard deviation of the training & cross validation data samples.
#   test_samples = amount of data samples for classification test.
#   test_clusterSTD = standard deviation of the test data samples.

alpha_param = 0.0001
lambda_param = 0.01
n_iterations = 15000

training_cv_samples = 500
training_cv_clusterStd = 2

test_samples = 150
test_clusterStd = 5

