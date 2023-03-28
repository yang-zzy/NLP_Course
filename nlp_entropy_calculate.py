import math
import os
import re
from collections import Counter

import jieba


def read_txt(path):
    txt_data = []
    files = os.listdir(path)
    for file in files:
        if 'inf' in file:  # 不读inf.txt,原文件夹中的网页文件被我删了
            continue
        file_path = os.path.join(path, file)
        with open(file_path, 'r', encoding='ANSI') as f:
            # 按自然段读取，将每段前后的换行去掉，同时删除原txt文件自带的开头与结尾
            text = [line.strip("\n").replace("\u3000", "").replace("\t", "") for line in f][3:-3]
            txt_data += text
        f.close()
    return txt_data, files


def txt_process(txt):
    # 去掉停词表中的词语
    # # with open('./cn_stopwords.txt', 'r', encoding='utf-8') as f:  # 停词表
    # with open('./cn_punctuation.txt', 'r', encoding='utf-8') as f:  # 仅删除标点符号
    #     stopwords = [line.strip() for line in f.readlines()]
    #     stopwords.append(' ')
    stopwords = []  # 考虑全文
    output = []
    for i in range(len(txt)):
        txt[i] = re.sub(r'[a-zA-Z0-9=]+', '', txt[i])  # 处理文中的英文及数字
        for word in jieba.cut(txt[i]):
            if word not in stopwords:
                output.append(word)
    # with open("stop_text.txt", "w", encoding="utf-8") as f:
    # with open("puc_deleted.txt", "w", encoding="utf-8") as f:
    with open("full_text.txt", "w", encoding="utf-8") as f:
        for i in output:
            f.write(i)
    return output


def one_word(file):
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
        word_list = [word for word in content]
    token_num = len(word_list)
    counter = Counter(word_list)
    words_table = counter.most_common()
    entropy_1gram = sum([-(i[1] / token_num) * math.log((i[1] / token_num), 2) for i in words_table])
    print("词库总词数：", token_num, " ", "不同词的个数：", len(words_table))
    print("出现频率前10的1-gram词语：", words_table[:10])
    print("entropy_1gram:", entropy_1gram)



def one_gram(token):
    print('One_gram')
    token_num = len(token)
    counter = Counter(token)
    words_table = counter.most_common()
    entropy_1gram = sum([-(i[1] / token_num) * math.log((i[1] / token_num), 2) for i in words_table])
    print("词库总词数：", token_num, " ", "不同词的个数：", len(words_table))
    print("出现频率前10的1-gram词语：", words_table[:10])
    print("entropy_1gram:", entropy_1gram)


def bigram(token):
    print('Bi_gram')
    token_bigram = []
    for i in range(len(token) - 1):
        token_bigram.append(token[i] + ' ' + token[i + 1])
    # 二元词袋的频率统计
    token_bigram_num = len(token_bigram)
    counter2 = Counter(token_bigram)
    words_table2 = counter2.most_common()
    print(words_table2[:20])
    # 2-gram的相同句首的频率统计，用于计算条件概率
    same_1st_word = [i.split(' ')[0] for i in token_bigram]
    assert token_bigram_num == len(same_1st_word)
    counter_1st = Counter(same_1st_word)
    words_table_1st = dict(counter_1st.most_common())
    entropy_bigram = 0
    for i in words_table2:
        p_xy = i[1] / token_bigram_num  # p(x,y)
        first_word = i[0].split(' ')[0]
        p_y = i[1] / words_table_1st[first_word]  # p(x|y)
        entropy_bigram += -p_xy * math.log(p_y, 2)
    print("词库总词数：", token_bigram_num, " ", "不同词的个数：", len(words_table2))
    print("entropy_2gram:", entropy_bigram)


def trigram(token):
    print('Tri_gram')
    token_trigram = []
    for i in range(len(token) - 2):
        token_trigram.append(token[i] + token[i + 1] + ' ' + token[i + 2])
    # 二元词袋的频率统计
    token_trigram_num = len(token_trigram)
    counter3 = Counter(token_trigram)
    words_table3 = counter3.most_common()
    print(words_table3[:20])
    # 2-gram的相同句首的频率统计，用于计算条件概率
    same_2nd_word = [i.split(' ')[0] for i in token_trigram]
    assert token_trigram_num == len(same_2nd_word)
    counter_2nd = Counter(same_2nd_word)
    words_table_2nd = dict(counter_2nd.most_common())
    entropy_trigram = 0
    for i in words_table3:
        p_xyz = i[1] / token_trigram_num  # p(x,y,z)
        first_2word = i[0].split(' ')[0]
        p_yz = i[1] / words_table_2nd[first_2word]  # p(x|y,z)
        entropy_trigram += -p_xyz * math.log(p_yz, 2)
    print("词库总词数：", token_trigram_num, " ", "不同词的个数：", len(words_table3))
    print("entropy_3gram:", entropy_trigram)


# 文本读取与预处理
# txt_path = './testtxt'
txt_path = './jyxstxtqj_downcc.com'
txt_data, files = read_txt(txt_path)
token_list = txt_process(txt_data)

one_word('./stop_text.txt')
one_word('./puc_deleted.txt')
one_word('./full_text.txt')
one_gram(token_list)
bigram(token_list)
trigram(token_list)
