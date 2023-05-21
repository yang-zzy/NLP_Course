import os

import config
from dataset import read_txt


def evaluate(corpus, file, models):
    file_list = []
    for root, path, fil in os.walk(file):
        for txt_file in fil:
            file_list.append(root + txt_file)
    seg_l = []
    for path in file_list:
        print(path)
        seg = read_txt(path)
        seg_l += seg
    for seg in seg_l:
        test, test_length = corpus.word2idx([seg])
        decoded_incdices = models.evaluation(test.to(config.device),test_length)
        result = []
        for i in decoded_incdices:
            result.append(corpus.dictionary.idx2word[i])
        print("".join(result))
    # return result
