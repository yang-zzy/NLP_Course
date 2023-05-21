import torch.nn as nn
import config
from dataset import vocab_size


class NumEncoder(nn.Module):
    def __init__(self):
        super(NumEncoder,self).__init__()
        self.vocab_size = vocab_size
        self.dropout = config.dropout
        self.embedding_dim = config.embedding_dim
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,embedding_dim=self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_layers,
                            batch_first=True,
                            dropout=config.dropout)

    def forward(self, input,input_length):
        embeded = self.embedding(input)
        embeded = nn.utils.rnn.pack_padded_sequence(embeded, lengths=input_length, batch_first=True,enforce_sorted=False)

        # hidden:[1,batch_size,vocab_size]
        out, hidden = self.lstm(embeded)
        out, outputs_length = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # out [batch_size,seq_len,hidden_size]
        # hidden [1,batch_size,hidden_size]
        return out, hidden