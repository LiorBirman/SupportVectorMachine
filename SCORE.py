from sklearn.metrics import f1_score


act_pos = [1 for _ in range(100)]
act_neg = [0 for _ in range(10000)]
y_true = act_pos + act_neg

# define predictions
pred_pos = [0 for _ in range(5)] + [1 for _ in range(95)]
pred_neg = [1 for _ in range(55)] + [0 for _ in range(9945)]
y_pred = pred_pos + pred_neg

# calculate score
score = f1_score(y_true, y_pred, average='binary')
print('F-Measure: %.3f' % score)
