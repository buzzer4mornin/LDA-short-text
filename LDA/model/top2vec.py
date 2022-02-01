with open("../input-data/eng_comments.txt", 'r') as f:
    lines = f.readlines()
    eng_comments = []
    for l in lines:
        eng_comments.append(l.split("\n")[0])

model = Top2Vec(eng_comments)
