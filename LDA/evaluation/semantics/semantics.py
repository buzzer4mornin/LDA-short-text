import numpy as np
from random import randrange

theta = np.load("theta.npy")
print(theta.shape)
with open("eng_comments.txt", 'r', encoding='utf-8') as f:
    eng_comments = f.readlines()


def random_picker(min_length, eng_comments, theta):
    while True:
        i = randrange(len(eng_comments))
        if len(eng_comments[i].split(" ")) > min_length:
            print(eng_comments[i])
            print(theta[i, :])
            break


random_picker(15, eng_comments, theta)





