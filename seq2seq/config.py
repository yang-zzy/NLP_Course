import torch

max_len = 50  # 文本序列长度
batch_size = 64
lr = 0.002

epochs = 50
dropout = 0.1
embedding_dim = 256
hidden_size = 128  # lstm隐层大小
num_layers = 2  # lstm层数

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
