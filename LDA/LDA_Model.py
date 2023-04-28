import os

import jieba
import matplotlib.pyplot as plt
import numpy as np
from gensim import corpora, models
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def read_txt(path: str, word_cut=True):
    """
    读取txt文件夹，并划分训练集与测试集
    :param path: 文本文件夹路径
    :param word_cut: 是否进行分词
    :return: 训练集与测试集的词列
    """
    files = os.listdir(path)
    train = {}
    test = {}
    for file in files:
        file_path = os.path.join(path, file)
        with open(file_path, 'r', encoding='ANSI') as f:
            con = f.read()
            con = content_deal(con)
            if word_cut:
                con = jieba.lcut(con)  # 结巴分词
            with open('E:/研究生课程/NLP/entropy_calculate/cn_stopwords.txt', 'r', encoding='utf-8') as stop:  # 删除停用词
                stopwords = [line.strip() for line in stop.readlines()]
                stopwords.append(' ')
            stop.close()
            stw_delete = []
            for word in con:
                if word not in stopwords:
                    stw_delete.append(word)
            con_list = list(stw_delete)
            pos = int(len(stw_delete) // 13)  ####16篇文章，分词后，每篇均匀选取13个500词段落进行建模
            train[file] = []
            test[file] = []
            for i in range(13):
                train[file].append(con_list[i * pos:i * pos + 500])
                test[file].append(con_list[i * pos + 500:i * pos + 1000])
        print(file)
        f.close()
    return train, test


def content_deal(content):  # 语料预处理，进行断句，去除标点符号和一些无意义内容
    with open('E:/研究生课程/NLP/entropy_calculate/cn_punctuation.txt', 'r', encoding='utf-8') as f:
        delete_words = [line.strip() for line in f.readlines()]
        delete_words.append(' ')
    delete_words += ['本书来自www', 'cr173', 'com', '免费txt小说下载站', '更多更新免费电子书请关注www',
                     '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
                     '\u3000', '\n', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b', '=', ',', '~', '\t']
    for a in delete_words:
        content = content.replace(a, '')
    return content


def lda_train(data_dict_train, data_dict_test, num_topics=15, passes=50, random_seed=42):
    # 将数据集转换为gensim需要的格式
    np.random.seed(random_seed)
    # 将数据集转换为gensim需要的格式
    labels = list(data_dict_train.keys())
    train_doc_list = []
    test_doc_list = []
    train_label_list = []
    test_label_list = []

    le = LabelEncoder()
    le.fit(labels)
    for label, paragraphs in data_dict_train.items():
        for para in paragraphs:
            train_doc_list.append(para)
            train_label_list.append(le.transform([label])[0])
    for label, paragraphs in data_dict_test.items():
        for para in paragraphs:
            test_doc_list.append(para)
            test_label_list.append(le.transform([label])[0])
    assert len(train_label_list) == len(train_doc_list) and len(test_label_list) == len(test_doc_list)

    # 创建词袋模型
    dictionary = corpora.Dictionary(train_doc_list + test_doc_list)

    # 将文本转换为词袋向量
    corpus_train = [dictionary.doc2bow(doc) for doc in train_doc_list]
    corpus_test = [dictionary.doc2bow(doc) for doc in test_doc_list]
    # 训练LDA模型
    lda_model = models.ldamodel.LdaModel(corpus_train, num_topics=num_topics, id2word=dictionary, passes=passes)

    # 输出模型的前20重要的topic中前10的词汇
    for topic in lda_model.print_topics():
        print(topic)

    # 将每个段落表示为主题分布
    topics_train = lda_model.get_document_topics(corpus_train, minimum_probability=0)
    topics_test = lda_model.get_document_topics(corpus_test, minimum_probability=0)
    train_topics = []
    test_topics = []
    for topics in topics_train:
        train_topics.append([k[1] for k in topics])
    for topics in topics_test:
        test_topics.append([k[1] for k in topics])
    # 使用SVM分类器进行分类

    svm_model = SVC(kernel='rbf', )
    svm_model.fit(train_topics, train_label_list)

    train_pred = svm_model.predict(train_topics)
    test_pred = svm_model.predict(test_topics)
    accuracy_train = svm_model.score(train_topics, train_label_list)
    accuracy_test = svm_model.score(test_topics, test_label_list)

    print(list(train_pred))
    print(list(test_pred))
    print('Train Accuracy:{:.4%}'.format(accuracy_train))
    print('Test Accuracy:{:.4%}'.format(accuracy_test))

    return accuracy_train, accuracy_test


if __name__ == "__main__":
    num_topics_list = [1, 5, 10, 20, 50, 100, 200, 500]
    passes_list = [1000, 200, 100, 50, 20, 10, 5, 2]
    # num_topics_list = [16]
    # passes_list = [50]
    data_dict_train, data_dict_test = read_txt('./jyxstxtqj_downcc.com', word_cut=False)
    train_acc_l = []
    test_acc_l = []
    for i in range(len(num_topics_list)):
        train_acc, test_acc = lda_train(data_dict_train, data_dict_test, num_topics=num_topics_list[i],
                                        passes=passes_list[i], random_seed=3407)
        train_acc_l.append(train_acc)
        test_acc_l.append(test_acc)
    x = np.arange(len(num_topics_list))
    plt.figure(1)
    plt.plot(x, train_acc_l)
    plt.xlabel('num_topics')
    plt.xticks(x, num_topics_list)
    plt.ylabel('acc')
    plt.title('train_acc')
    plt.figure(2)
    plt.plot(x, test_acc_l)
    plt.xlabel('num_topics')
    plt.xticks(x, num_topics_list)
    plt.ylabel('acc')
    plt.title('test_acc')
    plt.show()
