import os
import time
import pandas as pd
import numpy as np
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
    comments = [c for c in df_raw.columns if "TOPICS" not in c]
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
            if detect(comment) == "en":
                eng_comments.append(comment)
        except:
            continue

    return eng_comments


if __name__ == '__main__':
    eng_comments = get_vocabulary()
    print(eng_comments[3])
    print(eng_comments[5])