import random

import torch
import torch.nn as nn

import config
from dataset import vocab_size


class NumDecoder(nn.Module):
    def __init__(self):
        super(NumDecoder, self).__init__()
        self.max_seq_len = config.max_len
        self.vocab_size = vocab_size
        self.embedding_dim = config.embedding_dim
        self.dropout = config.dropout

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_layers,
                            batch_first=True,
                            dropout=self.dropout)

        self.fc = nn.Linear(config.hidden_size, self.vocab_size)

    def forward(self, encoder_hidden, target, target_length):

        decoder_input = torch.tensor([[0]] * config.batch_size, device=config.device)
        decoder_input = decoder_input.long()
        decoder_outputs = torch.zeros(config.batch_size, config.max_len, self.vocab_size,
                                      device=config.device)

        decoder_hidden = encoder_hidden

        for t in range(config.max_len):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[:, t, :] = decoder_output_t

            use_teacher_forcing = random.random() > 0.5
            if use_teacher_forcing:
                decoder_input = target[:, t].unsqueeze(1)  # [batch_size,1]
            else:
                value, index = torch.topk(decoder_output_t, 1)  # index [batch_size,1]
                decoder_input = index
        return decoder_outputs, decoder_hidden

    def forward_step(self, decoder_input, decoder_hidden):
        """
        :param decoder_input:[batch_size,1]
        :param decoder_hidden: [1,batch_size,hidden_size]
        :return: out:[batch_size,vocab_size],decoder_hidden:[1,batch_size,didden_size]
        """
        embeded = self.embedding(decoder_input)  # embeded: [batch_size,1 , embedding_dim]
        out, decoder_hidden = self.lstm(embeded, decoder_hidden)  # out [1, batch_size, hidden_size]
        out = out.squeeze(0)
        out = self.fc(out)
        out = out.squeeze(1)
        return out, decoder_hidden

    def evaluation(self, encoder_hidden):  # [1, 20, 14]
        # target = target.transpose(0, 1)  # batch_first = False
        batch_size = encoder_hidden[0].size(1)

        decoder_input = torch.tensor([[0] * batch_size], device=config.device).permute(1, 0)
        decoder_input = decoder_input.long()
        decoder_outputs = torch.zeros(batch_size, config.max_len, self.vocab_size,
                                      device=config.device)  # [seq_len,batch_size,14]
        decoder_hidden = encoder_hidden

        for t in range(config.max_len):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[:, t, :] = decoder_output_t
            value, index = torch.topk(decoder_output_t, 1)  # index [20,1]
            decoder_input = index

        # # 获取输出的id
        decoder_indices = []
        for i in range(decoder_outputs.size(1)):
            value, indices = torch.topk(decoder_outputs[:, i, :], 3)
            decoder_indices.append(int(indices[0][random.randint(0, 2)].data))
        # print(decoder_indices)
        return decoder_indices
