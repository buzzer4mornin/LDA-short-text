import os
import time
import pandas as pd
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re

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
    comments = [c for c in df_raw.columns if "COMMENT" in c]
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
    # terms = [w for w in terms if w not in stop_words and "_" not in w]  # remove words with underscore
    # disregard = ["work", "working", "workings", "people", "company", "merck", "msd", "employ", "peop"]
    # terms = [w for w in terms if not any(d in w for d in disregard)]
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
    print('Execution time: {:.2f} min'.format((end - start)/60))
    return vocab, eng_comments


def get_input_docs(vocab, eng_comments):
    """
    Create input text file for LDA
    """
    if os.path.exists("docs.txt"):
        os.remove("docs.txt")
    start = time.time()
    term_vs_index = {v_term: v_index for v_term, v_index in zip(vocab, range(len(vocab)))}
    stop_words = set(stopwords.words('english'))
    for comment in eng_comments:
        # TODO: ~~~ SEPARATOR ~~~
        # flag = False
        # if "COLUMN ENDS RIGHT HERE" in plt:
        #    flag = True
        # -----------------------------------
        comment = comment.lower()
        comment = re.sub(r"\S*\d\S*", "", comment).strip()  # remove words with numbers
        comment = RegexpTokenizer(r'\w{4,}').tokenize(comment)  # remove words of TODO: length < 3/4
        comment = [w for w in comment if w not in stop_words]  # remove stop-words
        # terms = [t for t in terms if t not in stop_words and "_" not in t]  # remove words with underscore
        # disregard = ["work", "working", "workings", "people", "company", "merck", "msd", "employ", "peop"]
        # terms = [w for w in terms if not any(d in w for d in disregard)]

        term_counts = {}
        for term in comment:
            try:
                term_counts[term_vs_index[term]] += 1
            except KeyError:
                term_counts[term_vs_index[term]] = 1
        unique_terms = len(term_counts.keys())
        if unique_terms == 0:
            continue
        term_counts = str(term_counts).replace("{", "").replace("}", "").replace(" ", "").replace(",", " ")
        with open("docs.txt", 'a', encoding='utf-8') as f:
            f.write(str(unique_terms) + " " + term_counts + "\n")
            # TODO: ~~~ SEPARATOR ~~~
            # if flag:
            #    print("yes")
            #    f.write(f"{plt} \n")
            # else:
            #    f.write(str(unique_terms) + " " + term_counts + "\n")
            # --------------------------------------------------------
    end = time.time()
    print('\n-*-*-* Successfully Created "docs.txt" *-*-*-')
    print("Execution time: {:.2f} min".format((end - start)/60))


if __name__ == '__main__':
    vocab, eng_comments = get_vocabulary()
    print(len(eng_comments))
    #get_input_docs(vocab, eng_comments)
