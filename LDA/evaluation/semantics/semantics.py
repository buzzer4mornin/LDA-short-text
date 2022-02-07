import numpy as np
from random import randrange

global theta, eng_comments, auto_topics

theta = np.load("./../../output-data/theta.npy")
print(theta.shape)

with open("./../../input-data/eng_comments.txt", 'r', encoding='utf-8') as f:
    eng_comments = f.readlines()
with open("./../../output-data/topn_output.txt", 'r', encoding='utf-8') as f:
    auto_topics = f.readlines()


# theta = np.load("./../../saved-outputs/showcase_2/theta.npy")
# print(theta.shape)
#
# with open("./../../saved-outputs/showcase_2/eng_comments.txt", 'r', encoding='utf-8') as f:
#    eng_comments = f.readlines()
# with open("./../../saved-outputs/showcase_2/topn_output.txt", 'r', encoding='utf-8') as f:
#    auto_topics = f.readlines()

def random_picker(min_length, max_length):
    while True:
        i = randrange(len(eng_comments))
        if min_length < len(eng_comments[i].split(" ")) < max_length:
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


random_picker(min_length=10, max_length=35)


def topic_picker(which_topic, min_prop, max_length):
    which_topic -= 1  # changing into index based
    while True:
        i = randrange(len(eng_comments))
        if len(eng_comments[i].split(" ")) > max_length and np.array(theta[i, :])[which_topic] * 100 > min_prop:
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

#topic_picker(which_topic=5, min_prop=60, max_length=10)
