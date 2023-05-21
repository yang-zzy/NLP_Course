import os
import re

import jieba
import torch
from torch.utils.data import Dataset, DataLoader

import config


def read_txt(path: str):
    """
    读取txt文件夹
    :param path: 文本文件夹路径
    :return: 训练集与测试集的词列
    """
    with open(path, 'r', encoding='ANSI') as f:
        con = f.read()
        con = content_deal(con)
        seg_l = []
        cut = jieba.lcut(con)
        for i in range(len(cut) // config.max_len):
            seg_l.append(cut[i:i + config.max_len])
    f.close()
    return seg_l


def content_deal(content):  # 语料预处理，去除一些无意义内容
    delete_words = ['本书来自www', 'cr173', 'com', '免费txt小说下载站', '更多更新免费电子书请关注www',
                    '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
                    '\u3000', '\n', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b', '=', '~', '\t', ]
    for a in delete_words:
        content = content.replace(a, '')
        content = re.sub('\.', '', content)
        content = re.sub('\s', '', content)
    return content


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx  # 以字为键
            self.idx2word[self.idx] = word  # 以数值为键
            self.idx += 1


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()  # 继承类，初始化映射表
        self.file_list = []

    def get_file(self, filepath):
        for root, path, fil in os.walk(filepath):
            for txt_file in fil:
                self.file_list.append(root + txt_file)
        return self.file_list

    def get_data(self):  # 读取文件，导入映射表
        # step 1
        seg_l = []
        for path in self.file_list:
            print(path)
            seg = read_txt(path)
            for line in seg:
                for word in line:
                    self.dictionary.add_word(word)
            seg_l += seg
        return self.word2idx(seg_l)

    def word2idx(self, seg_l):
        ids = torch.zeros((len(seg_l), config.max_len),
                          dtype=torch.int32)  # 实例化一个LongTensor，命名为ids。遍历全部文本，根据映射表把单词转成索引，存入ids里
        lenth = torch.LongTensor(len(seg_l))
        for i, seg in enumerate(seg_l):
            for j, word in enumerate(seg):
                try:
                    ids[i][j] = self.dictionary.word2idx[word]
                except KeyError:
                    ids[i][j] = self.dictionary.word2idx['。']  # 遇到分词出现不一致导致字典不对就替换为句号
            lenth[i] = len(seg)
        return ids, lenth

    def idx2word(self, idx_l):
        for i, idx in enumerate(idx_l):
            for j, word in enumerate(idx):
                idx_l[i][j] = self.dictionary.idx2word[word]
        return idx_l


class NoveDataset(Dataset):
    def __init__(self, device):
        super(NoveDataset, self).__init__()
        self.corpus = Corpus()
        self.corpus.get_file('E:\\研究生课程\\NLP\\entropy_calculate\\testtxt\\')
        self.ids, self.length = self.corpus.get_data()
        self.ids = self.ids.to(device)
        self.vocab_size = len(self.corpus.dictionary)

    def __getitem__(self, index):
        return self.ids[index], self.ids[index + 1], self.length[index], self.length[index + 1]

    def __len__(self):
        return len(self.ids) - 1


dataset = NoveDataset(config.device)
vocab_size = dataset.vocab_size
data_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, drop_last=True)
if __name__ == '__main__':
    for idx, (input, target, input_length, target_length) in enumerate(data_loader):
        print(idx)
        print(input)
        print(target)
        print(input_length)
        print(target_length)
        break
