import numpy as np

predicted = np.ones(5)
cv = np.zeros(5)

cv[0] = 1
cv[3] = 1

print("a, b", predicted, cv)

asd = predicted * 10 + cv

true_positive = sum(asd == 11)

print(true_positive)
