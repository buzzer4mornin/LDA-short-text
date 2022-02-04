import sys
import numpy as np
import per_vb
import per_fw
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


def read_data_for_perpl(test_data_folder):
    filename_part1 = f'{test_data_folder}/docs_test_part_1.txt'
    filename_part2 = f'{test_data_folder}/docs_test_part_2.txt'
    wordids_1, wordcts_1 = read_data(filename_part1)
    wordids_2, wordcts_2 = read_data(filename_part2)
    return wordids_1, wordcts_1, wordids_2, wordcts_2


def compute_perplexities_vb(beta, alpha, eta, max_iter, wordids_1, wordcts_1, wordids_2, wordcts_2):
    vb = per_vb.VB(beta, alpha, eta, max_iter)
    LD2 = vb.compute_perplexity(wordids_1, wordcts_1, wordids_2, wordcts_2)
    return LD2


def compute_perplexities_fw(beta, max_iter, wordids_1, wordcts_1, wordids_2, wordcts_2):
    fw = per_fw.FW(beta, max_iter)
    LD2 = fw.compute_perplexity(wordids_1, wordcts_1, wordids_2, wordcts_2)
    return LD2


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
        try:
            vals.append(float(set_val[1]))
        except:
            vals.append(str(set_val[1]))
    ddict = dict(zip(sets, vals))
    ddict['num_docs'] = int(ddict['num_docs'])
    ddict['num_words'] = int(ddict['num_words'])
    ddict['num_topics'] = int(ddict['num_topics'])
    ddict['batch_size'] = int(ddict['batch_size'])
    ddict['tops'] = int(ddict['tops'])
    ddict['alpha'] = float(ddict['alpha'])
    ddict['eta'] = float(ddict['eta'])
    ddict['tau0'] = int(ddict['tau0'])
    ddict['kappa'] = float(ddict['kappa'])
    ddict['BOPE'] = str(ddict['BOPE'])
    ddict['iter_infer'] = int(ddict['iter_infer'])
    ddict['iter_train'] = int(ddict['iter_train'])
    return ddict


def read_minibatch_list_frequencies(fp, batch_size):
    wordids = list()
    wordcts = list()
    for i in range(batch_size):
        line = fp.readline()
        # check end of file
        if len(line) < 5:
            break
        terms = str.split(line)
        doc_length = int(terms[0])
        ids = np.zeros(doc_length, dtype=np.int32)
        cts = np.zeros(doc_length, dtype=np.int32)
        for j in range(1, doc_length + 1):
            term_count = terms[j].split(':')
            ids[j - 1] = int(term_count[0])
            cts[j - 1] = int(term_count[1])
        wordids.append(ids)
        wordcts.append(cts)
    return wordids, wordcts


def compute_sparsity(doc_tp, batch_size, num_topics, _type):
    sparsity = np.zeros(batch_size, dtype=np.float)
    if _type == 'z':
        for d in range(batch_size):
            N_z = np.zeros(num_topics, dtype=np.int)
            N = len(doc_tp[d])
            for i in range(N):
                N_z[doc_tp[d][i]] += 1.
            sparsity[d] = len(np.where(N_z != 0)[0])
    else:
        for d in range(batch_size):
            sparsity[d] = len(np.where(doc_tp[d] > 1e-10)[0])
    sparsity /= num_topics
    return np.mean(sparsity)


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


def print_diff_list_tops_v2(list_tops, prev_list_tops, i):
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


def print_diff_list_tops(list_tops, prev_list_tops):
    top_ns = len(list_tops[0])
    diff_count = 0
    for i, j in zip(prev_list_tops, list_tops):
        diff_count += top_ns - len(list(set(i) & set(j)))
    print("Difference:", diff_count)


def print_topics(vocab_file, nwords, result_file):
    with open(vocab_file, 'r') as f:
        vocab = f.readlines()

    vocab = list(map(lambda x: x.strip(), vocab))
    vocab_index = {i: w for i, w in zip(range(len(vocab)), vocab)}

    for l in nwords:
        converts = list(map(lambda x: vocab_index[int(x)], l))
        converts = " ".join(converts)
        with open(result_file, "a") as r:
            r.write(converts + "\n")


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

    vocab_file = "./input-data/vocab.txt"
    result_file = f"{output_folder}/topn_output.txt"
    print_topics(vocab_file, list_tops, result_file)
