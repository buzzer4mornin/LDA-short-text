with open("nyt_50k.txt", 'r', encoding='utf-8') as f:
    train = f.readlines()

with open("data_test_2_part_1.txt", 'r', encoding='utf-8') as f:
    test_1 = f.readlines()

with open("data_test_2_part_2.txt", 'r', encoding='utf-8') as f:
    test_2 = f.readlines()


print(test_2[5])
print(test_1[5])