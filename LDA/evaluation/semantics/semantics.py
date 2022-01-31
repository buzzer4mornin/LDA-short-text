import numpy as np
from random import randrange

global theta, eng_comments, auto_topics, manual_topics

theta = np.load("theta.npy")
# print(theta.shape)

with open("eng_comments.txt", 'r', encoding='utf-8') as f:
    eng_comments = f.readlines()
with open("topn_output.txt", 'r', encoding='utf-8') as f:
    auto_topics = f.readlines()

manual_topics = ["management", "politics/products", "customers/sales", "org. process/barrier", "org. infrastructure",
                 "diversity/inclusion",
                 "remote + covid", "products", "touches many topics???", "work-life bal. (covid+work)",
                 "emp+talent+manager",
                 "work-life bal. (resource/support)", "feel about merck / mission", "working workload", "career"]


def random_picker(min_length):
    while True:
        i = randrange(len(eng_comments))
        if len(eng_comments[i].split(" ")) > min_length:
            print(f"\nCOMMENT:\n{eng_comments[i]}")
            # print(theta[i, :])
            topic_props = np.array(theta[i, :]) * 100
            for i in range(len(topic_props)):
                topic_props[i] = round(topic_props[i], 2)
            best_topics = np.where(topic_props > 0)[0]
            for best in best_topics:
                print("-------------------------")
                print(f"{topic_props[best]}% \n{auto_topics[best]}")
            break


random_picker(15)


def topic_picker(which_topic, min_prop, min_length):
    which_topic -= 1  # changing into index based
    while True:
        i = randrange(len(eng_comments))
        if len(eng_comments[i].split(" ")) > min_length and np.array(theta[i, :])[which_topic] * 100 > min_prop:
            print(f"\nCOMMENT:\n{eng_comments[i]}")
            # print(theta[i, :])
            topic_props = np.array(theta[i, :]) * 100
            for i in range(len(topic_props)):
                topic_props[i] = round(topic_props[i], 2)
            best_topics = np.where(topic_props > 0)[0]
            for best in best_topics:
                print("-------------------------")
                print(f"{manual_topics[best]} -- {topic_props[best]}% \n{auto_topics[best]}")
            break


# topic_picker(which_topic=13, min_prop=90, min_length=10)

# To be excluded
# Barriers to Execution_COMMENT_TOPICS
# Resources_Comment_Topics - CA
# Next steps_comments_topics - BR
# Prospects_COMMENT_TOPICS -  BU