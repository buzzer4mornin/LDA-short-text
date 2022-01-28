import sys
import math
import numpy as np
import os

#t = np.load("./theta.npy")
#t = t[33736:36583, :]
#t = t.sum(axis=0)/t.shape[0]
#print(t*100)

t = np.load("./theta.npy")
print(t.shape)