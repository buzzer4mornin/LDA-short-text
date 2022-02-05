import os
import time
import pandas as pd
import numpy as np
import pickle
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
import random


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
    comments = [c for c in df_raw.columns if "TOPICS" in c]
    df_comments = df_raw[[*comments]]

    # filter out null comment columns
    notnull_cols = []
    for col in df_comments.columns:
        if df_comments[col].notnull().sum() != 0:
            notnull_cols.append(col)
    df_comments = df_comments[[*notnull_cols]]

    # ----------------------------
    # df_comments = df_comments["Barriers to Execution_COMMENT_TOPICS"]        # <<---------------- #TODO: Subselecting the column
    # ----------------------------

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
    term_counts = {}
    for comment in all_comments:
        try:
            if detect(comment) == "en":
                eng_comments.append(comment)
                for i in comment.split(", "):
                    try:
                        term_counts[i] += 1
                    except KeyError:
                        term_counts[i] = 1
        except:
            continue
    with open('term_counts.pkl', 'wb') as f:
        pickle.dump(term_counts, f)


if __name__ == '__main__':
    with open('term_counts.pkl', 'rb') as f:
         term_counts = pickle.load(f)
    new = dict(sorted(term_counts.items(), key=lambda item: item[1], reverse=True))
    print(new)
    #get_vocabulary()


# Top Concerns --> {'Staying Healthy': 27388, 'Personal Safety': 5780, 'Job Security': 16281, 'Politics': 4903, 'Child/Family Care': 16029, 'Other (please specify)': 4750, 'Receiving Most Current Updates': 3398, 'Racial Injustice': 3342, 'Work Resources': 13265}
