import pandas as pd
import numpy as np

with open("list_tops.txt", 'r', encoding='utf-8') as f:
    list_tops = []
    for line in f:
        l = line.split(" ")
        l[-1] = l[-1].split("\n")[0]
        list_tops.append(l)

with open("prev_list_tops.txt", 'r', encoding='utf-8') as f:
    prev_list_tops = []
    for line in f:
        l = line.split(" ")
        l[-1] = l[-1].split("\n")[0]
        prev_list_tops.append(l)


def find_diff(prev_list_tops, list_tops):
    top_n_size = len(list_tops[0])
    diff = 0
    for i, j in zip(prev_list_tops, list_tops):
        diff += top_n_size - len(list(set(i) & set(j)))
    return diff


find_diff(prev_list_tops, list_tops)

# print(list_tops)
# print(prev_list_tops)
