import sys
import numpy as np

def list_top(beta, tops):
    min_float = -sys.float_info.max
    num_tops = beta.shape[0]
    list_tops = list()
    for k in range(num_tops):
        top = list()
        arr = np.array(beta[k, :], copy=True)
        for t in range(tops):
            index = arr.argmax()
            top.append(arr[index])
            arr[index] = min_float
        #list_tops.append(top)
        print(round(sum(top), 2), np.array(top)/sum(top))
    return list_tops

if __name__ == '__main__':
    beta = np.load("../../output-data/beta.npy")
    # theta = np.load("../../output-data/theta.npy")
    # print(theta.sum(axis=0)/sum(theta.sum(axis=0)))
    result_file = "./topn_output_frequency.txt"
    list_top(beta, 20)