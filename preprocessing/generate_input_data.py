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
    if os.path.exists("../LDA/input-data/vocab.txt"):
        os.remove("../LDA/input-data/vocab.txt")

    start = time.time()

    # read raw data
    df_raw = pd.read_csv("pulse_q2_2021_raw.csv")
    # df_raw = df_raw.iloc[350:450, :] ! for test purposes - subselecting rows

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
        all_comments += [*list_col]

    # extract english comments
    eng_comments = []
    for comment in all_comments:
        try:
            if detect(comment) == "en":  # and len(comment.split(" ")) > 15:
                eng_comments.append(comment)
        except:
            continue

    # merge corpus and clean it
    terms = ' '.join(eng_comments)
    terms = cleaner(terms)
    terms = set(terms)

    # extract Vocabulary
    vocab = sorted(set(list(terms)))
    with open("../LDA/input-data/vocab.txt", 'w', encoding='utf-8') as f:
        for term in vocab:
            f.write(term + "\n")

    end = time.time()
    print('-*-*-* Successfully Created "vocab.txt" *-*-*-')
    print("Total number of english comments:", len(eng_comments))
    print("Vocabulary size:", len(vocab))
    print('Execution time: {:.2f} min'.format((end - start) / 60))
    return vocab, eng_comments


def cleaner(obj):
    stop_words = set(stopwords.words('english'))
    obj = obj.lower()
    obj = re.sub(r"\S*\d\S*", "", obj).strip()  # remove words with numbers
    obj = RegexpTokenizer(r'\w{4,}').tokenize(obj)  # remove words of TODO: length < 4
    obj = [o for o in obj if o not in stop_words]  # remove stop-words
    disregard = ["work", "working", "workings", "people", "company", "merck", "msd", "employ", "peop"]
    obj = [o for o in obj if not any(d in o for d in disregard)]
    return obj


def write_comment(docs_dir, comment):
    term_vs_index = {v_term: v_index for v_term, v_index in zip(vocab, range(len(vocab)))}
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


def get_input_docs(test_size, test_obs_to_ho_ratio):
    """
    Create input text file for LDA
    """
    start = time.time()
    if os.path.exists("../LDA/input-data/docs.txt"):
        os.remove("../LDA/input-data/docs.txt")
    if os.path.exists("../LDA/input-data/docs_test_part_1.txt"):
        os.remove("../LDA/input-data/docs_test_part_1.txt")
    if os.path.exists("../LDA/input-data/docs_test_part_2.txt"):
        os.remove("../LDA/input-data/docs_test_part_2.txt")

    len_obs, len_ho = 0, 0
    test_indices = np.array(random.sample(range(len(eng_comments)), test_size))

    for i, comment in enumerate(eng_comments):
        # to save as raw comment (check end of loop)
        comment_copy = comment

        # comment cleaner
        comment = cleaner(comment)

        # pass the comment if it vanished after cleaning :)
        if len(comment) == 0:
            continue

        # --- these comments will be in test data
        if i in test_indices:
            comment_obs, comment_ho = [], []
            ho_indices = np.array(random.sample(range(len(comment)), int((len(comment) * test_obs_to_ho_ratio))))
            for c_index, c in enumerate(comment):
                if c_index in ho_indices:
                    comment_ho.append(c)
                else:
                    comment_obs.append(c)

            # if comment_ho gets 0 share out of comment, then push it into training data
            if len(comment_ho) == 0:
                write_comment("../LDA/input-data/docs.txt", comment)
                with open("../LDA/input-data/eng_comments.txt", 'a', encoding='utf-8') as f:
                    f.write(comment_copy + "\n")
            else:
                write_comment("../LDA/input-data/docs_test_part_2.txt", comment_obs)
                write_comment("../LDA/input-data/docs_test_part_1.txt", comment_ho)
                try:
                    len_ho += len(comment_ho)
                    len_obs += len(comment_obs)
                except:
                    pass

        # --- these comments will be in training data
        else:
            write_comment("../LDA/input-data/docs.txt", comment)
            # save raw comment for later analysis
            with open("../LDA/input-data/eng_comments.txt", 'a', encoding='utf-8') as f:
                f.write(comment_copy + "\n")

    end = time.time()
    print('\n-*-*-* Successfully Created docs *-*-*-')
    print("Resulting obs to who ratio:", round(len_ho / len_obs, 2))
    print("Execution time: {:.2f} min".format((end - start) / 60))


if __name__ == '__main__':
    vocab, eng_comments = get_vocabulary()
    get_input_docs(test_size=10, test_obs_to_ho_ratio=0.2)
