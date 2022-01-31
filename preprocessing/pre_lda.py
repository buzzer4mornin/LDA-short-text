import os
import time
import pandas as pd
import numpy as np
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
import random

"""
[M] [term_1]:[count] [term_2]:[count] ...  [term_N]:[count]
[M]     - number of unique words in plot
[term]  - integer which is index of the word in vocabulary
[count] - how many times each word appeared in the plot

"""


def get_vocabulary():
    """
    Create and return vocabulary out of descriptions
    """
    if os.path.exists("vocab.txt"):
        os.remove("vocab.txt")

    start = time.time()

    # read raw data
    df_raw = pd.read_csv("pulse_q2_2021_raw.csv")

    # find and extract comment columns
    comments = [c for c in df_raw.columns if "COMMENT" in c and "TOPICS" not in c]
    df_comments = df_raw[[*comments]]

    # filter out null comment columns
    notnull_cols = []
    for col in df_comments.columns:
        if df_comments[col].notnull().sum() != 0:
            notnull_cols.append(col)
    df_comments = df_comments[[*notnull_cols]]

    # extract comments from comment columns into all_comments list
    all_comments = []
    for col in df_comments.columns:
        list_col = df_comments[col]
        mask = []
        for i in list_col:
            if len(str(i)) == 3:
                mask.append(False)
            else:
                mask.append(True)
        list_col = list(list_col[mask])
        # TODO: ~~~ SEPARATOR ~~~
        # if len(list_c) > 0:
        #    list_c.append(f"COLUMN ENDS RIGHT HERE: {c}")
        # ------------------------------------------------
        all_comments += [*list_col]

    # extract english comments
    eng_comments = []
    for comment in all_comments:
        try:
            if detect(comment) == "en":  # and len(comment.split(" ")) > 15:
                eng_comments.append(comment)
        except:
            continue

    stop_words = set(stopwords.words('english'))
    terms = ' '.join(eng_comments).lower()
    terms = re.sub(r"\S*\d\S*", "", terms).strip()  # remove words with numbers
    terms = RegexpTokenizer(r'\w{4,}').tokenize(terms)  # remove words of TODO: length < 3/4
    terms = [w for w in terms if w not in stop_words]  # remove stop-words
    # terms = [t for t in terms if t not in stop_words and "_" not in w]  # remove words with underscore
    disregard = ["work", "working", "workings", "people", "company", "merck", "msd", "employ", "peop"]
    terms = [t for t in terms if not any(d in t for d in disregard)]
    terms = set(terms)

    # Create Vocabulary textfile
    vocab = sorted(set(list(terms)))
    with open("vocab.txt", 'w', encoding='utf-8') as f:
        for term in vocab:
            f.write(term + "\n")

    end = time.time()
    print('-*-*-* Successfully Created "vocab.txt" *-*-*-')
    print("Total number of english comments:", len(terms))
    print("Vocabulary size:", len(vocab))
    print('Execution time: {:.2f} min'.format((end - start) / 60))
    return vocab, eng_comments


def write_comment(docs_dir, eng_comment_dir,  comment, vocab):
    term_vs_index = {v_term: v_index for v_term, v_index in zip(vocab, range(len(vocab)))}
    stop_words = set(stopwords.words('english'))
    comment_readable = comment

    # TODO: ~~~ SEPARATOR ~~~
    # flag = False
    # if "COLUMN ENDS RIGHT HERE" in plt:
    #    flag = True
    # -----------------------------------
    comment = comment.lower()
    comment = re.sub(r"\S*\d\S*", "", comment).strip()  # remove words with numbers
    comment = RegexpTokenizer(r'\w{4,}').tokenize(comment)  # remove words of TODO: length < 3/4
    comment = [c for c in comment if c not in stop_words]  # remove stop-words
    # comment = [c for c in comment if t not in stop_words and "_" not in t]  # remove words with underscore
    disregard = ["work", "working", "workings", "people", "company", "merck", "msd", "employ", "peop"]
    comment = [c for c in comment if not any(d in c for d in disregard)]

    term_counts = {}
    for term in comment:
        try:
            term_counts[term_vs_index[term]] += 1
        except KeyError:
            term_counts[term_vs_index[term]] = 1
    unique_terms = len(term_counts.keys())
    if unique_terms == 0:
        return
    term_counts = str(term_counts).replace("{", "").replace("}", "").replace(" ", "").replace(",", " ")

    with open(docs_dir, 'a', encoding='utf-8') as f:
        f.write(str(unique_terms) + " " + term_counts + "\n")
        # TODO: ~~~ SEPARATOR ~~~
        # if flag:
        #    print("yes")
        #    f.write(f"{plt} \n")
        # else:
        #    f.write(str(unique_terms) + " " + term_counts + "\n")
        # --------------------------------------------------------
    with open(eng_comment_dir, 'a', encoding='utf-8') as f:
        f.write(comment_readable + "\n")


def get_input_docs(vocab, eng_comments, test_size, test_set_split_proportion):
    """
    Create input text file for LDA
    """
    if os.path.exists("docs.txt"):
        os.remove("docs.txt")
    start = time.time()

    test_indices = np.array(random.sample(range(len(eng_comments)), test_size))
    part_1_indices = np.array(random.sample(range(len(test_indices)), int(len(test_indices) * test_set_split_proportion)))
    test_indices_part_1 = test_indices[part_1_indices]
    test_indices_part_2 = np.setdiff1d(test_indices, test_indices_part_1)

    print(test_indices_part_1.shape, test_indices_part_2.shape)

    for i, comment in enumerate(eng_comments):
        if i in test_indices_part_1:
            write_comment("docs_test_part_1.txt", "eng_comments_test_part_1.txt", comment, vocab)
        elif i in test_indices_part_2:
            write_comment("docs_test_part_2.txt", "eng_comments_test_part_2.txt", comment, vocab)
        else:
            write_comment("docs.txt", "eng_comments.txt", comment, vocab)

    end = time.time()
    print('\n-*-*-* Successfully Created "docs.txt" *-*-*-')
    print("Execution time: {:.2f} min".format((end - start) / 60))


if __name__ == '__main__':
    vocab, eng_comments = get_vocabulary()
    get_input_docs(vocab, eng_comments, test_size=2000, test_set_split_proportion=0.75)