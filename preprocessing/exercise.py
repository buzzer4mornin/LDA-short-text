import pandas as pd
from langdetect import detect

# read raw data
df = pd.read_csv("pulse_q2_2021_raw.csv")

comments = [c for c in df.columns if "COMMENT" in c]

# extract comment columns only
df = df[[*comments]]

# filter null columns
notnull_cols = []
for col in df.columns:
    if df[col].notnull().sum() != 0:
        notnull_cols.append(col)
df = df[[*notnull_cols]]

# extract all comments into all_list
all_comments = []
for c in df.columns:
    list_c = df[c]
    mask = []
    for i in list_c:
        if len(str(i)) == 3:
            mask.append(False)
        else:
            mask.append(True)
    list_c = list_c[mask]
    all_comments += [*list_c]

# extract english comments
eng_comments = []
c = 0
d = 0
avg_len = 0
for index, comment in enumerate(all_comments):
    try:
        # if detect(comment) == "en" and len(comment.split(" ")) > 15:
        if detect(comment) == "en":
            avg_len += len(comment.split(" "))
            c += 1
            eng_comments.append(comment)
        # elif detect(comment) == "en" and len(comment.split(" ")) < 15:
        #    d += 1
        # print(index, "---", c, "---", d)
    except:
        continue

print(len(eng_comments)/len(all_comments))
print(avg_len/c)