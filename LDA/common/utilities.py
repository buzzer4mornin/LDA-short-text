import sys
import os
import numpy as np
# import numpy_indexed
import pandas as pd
import pickle
# from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings("ignore")


def list_top(beta, tops):
    min_float = -sys.float_info.max
    num_tops = beta.shape[0]
    list_tops = list()
    for k in range(num_tops):
        top = list()
        arr = np.array(beta[k, :], copy=True)
        for t in range(tops):
            index = arr.argmax()
            top.append(index)
            arr[index] = min_float
        list_tops.append(top)
    return list_tops


def read_data(filename):
    wordids = list()
    wordcts = list()
    fp = open(filename, 'r')
    while True:
        line = fp.readline()
        # check end of file
        if len(line) < 1:
            break
        terms = line.split(' ')
        doc_length = int(terms[0])
        ids = np.zeros(doc_length, dtype=np.int32)
        cts = np.zeros(doc_length, dtype=np.int32)
        for j in range(1, doc_length + 1):
            term_count = terms[j].split(':')
            ids[j - 1] = int(term_count[0])
            cts[j - 1] = int(term_count[1])
        wordids.append(ids)
        wordcts.append(cts)
    fp.close()
    return wordids, wordcts


def read_setting(file_name):
    f = open(file_name, 'r')
    settings = f.readlines()
    f.close()
    sets = list()
    vals = list()
    for i in range(len(settings)):
        if settings[i][0] == '#':
            continue
        set_val = settings[i].split(':')
        sets.append(set_val[0])
        vals.append(float(set_val[1]))
    ddict = dict(zip(sets, vals))
    ddict['num_docs'] = int(ddict['num_docs'])
    ddict['num_terms'] = int(ddict['num_terms'])
    ddict['num_topics'] = int(ddict['num_topics'])
    ddict['tops'] = int(ddict['tops'])
    ddict['iter_infer'] = int(ddict['iter_infer'])
    ddict['iter_train'] = int(ddict['iter_train'])
    return ddict


def write_topic_top(list_tops, file_name):
    num_topics = len(list_tops)
    tops = len(list_tops[0])
    f = open(file_name, 'w')
    for k in range(num_topics):
        for j in range(tops - 1):
            f.write('%d ' % (list_tops[k][j]))
        f.write('%d\n' % (list_tops[k][tops - 1]))
    f.close()


def write_setting(ddict, file_name):
    keys = list(ddict.keys())
    vals = list(ddict.values())
    f = open(file_name, 'w')
    for i in range(len(keys)):
        f.write(f"{keys[i]}: {vals[i]}\n")
    f.close()


def print_diff_list_tops(list_tops, prev_list_tops, i):
    if i == 0:
        num_topics = len(list_tops)
        tops = len(list_tops[0])
        list_tops = np.array(list_tops)
        init = np.negative(np.ones([num_topics, tops], dtype=int))
        diff = init == list_tops
        diff_count = np.count_nonzero(diff)
        print("Difference:", diff_count)
    else:
        list_tops = np.array(list_tops)
        diff = prev_list_tops == list_tops
        diff_count = np.count_nonzero(diff)
        print("Difference:", diff_count)


def write_file(output_folder, saved_outputs_folder, list_tops, algo):

    def write(folder):
        list_tops_file_name = f'{folder}/list_tops.txt'
        write_topic_top(list_tops, list_tops_file_name)
        files = [attr for attr in dir(algo) if attr in ["beta", "theta"]]

        def file_locator(x): return f'{folder}/{str(x)}'

        for file in files:
            np.save(file_locator(file), getattr(algo, file))

    write(output_folder)
    write(saved_outputs_folder)
