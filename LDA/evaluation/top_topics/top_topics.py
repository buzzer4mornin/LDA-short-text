import numpy as np
from random import randrange

theta = np.load("./../../output-data/theta.npy")

print(theta.shape)
#theta_x = theta[0:10, :]
print(theta.sum(axis=0)/sum(theta.sum(axis=0)))