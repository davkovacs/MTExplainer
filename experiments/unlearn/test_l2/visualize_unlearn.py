import numpy as np
import matplotlib.pyplot as plt

scores = np.load("scores.npy")  # scores of the untrained models
print("The number of reactions unlearnt so far is {}".format(len(scores)))
score_interpr = np.load("score_interpr.npy")  # score of the original model
score_diff = scores - score_interpr[0]
print(score_interpr)
print("The probability of the product is {0:.5g}".format(np.exp(-score_interpr[0])))
plt.hist(score_diff, bins=50)
plt.title("The histogram of changes in negative log-probs of product")
plt.show()

print("The line number of the points decreasing the probability the most is:")
max_ind = score_diff.argsort()[-5:]
print(max_ind)
print(score_diff[max_ind])