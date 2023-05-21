import os

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim

import config
from dataset import data_loader, dataset
from decoder import NumDecoder
from encoder import NumEncoder
from eval import evaluate
from seq2seq import Seq2Seq

encoder = NumEncoder()
decoder = NumDecoder()
model = Seq2Seq(encoder, decoder, config.device)
print(model)
print("\nInitializing weights...")
for name, param in model.named_parameters():
    if 'bias' in name:
        torch.nn.init.constant_(param, 0.0)
    elif 'weight' in name:
        torch.nn.init.xavier_normal_(param)

optimizer = optim.Adam(model.parameters(), lr=config.lr)
criterion = nn.CrossEntropyLoss(reduction="mean")


def get_loss(decoder_outputs, target):
    target = target.long()
    target = target.view(-1)  # [batch_size*max_len]
    decoder_outputs = decoder_outputs.view(config.batch_size * config.max_len, -1)
    return criterion(decoder_outputs, target)


def train(epoch):
    for idx, (input, target, input_length, target_len) in enumerate(data_loader):
        loss_l = []
        optimizer.zero_grad()
        ##[seq_len,batch_size,vocab_size] [batch_size,seq_len]
        decoder_outputs, decoder_hidden = model(input, target, input_length, target_len)
        loss = get_loss(decoder_outputs, target)
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        decoder_indices = []
        # decoder_outputs = decoder_outputs.transpose(0,1) #[batch_size,seq_len,vocab_size]
        # print("decoder_outputs size",decoder_outputs.size())
        for j in range(decoder_outputs.size(0)):
            de = []
            for i in range(decoder_outputs.size(1)):
                value, indices = torch.topk(decoder_outputs[j, i, :], 3)
                # print("indices size",indices.size(),indices)
                # indices  = indices.transpose(0,1)
                de.append(int(indices[0].data))
            decoder_indices.append(de)

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch + 1, idx * len(input), len(data_loader.dataset), 100. * idx / len(data_loader), loss.item()))
        loss_l.append(loss.item())
        if loss.item() < 1.5:
            print('loss<1.5, saving model...')
            torch.save(model.state_dict(), "model/seq2seq_model_best.pkl")
    return model, sum(loss_l) / len(loss_l)


if __name__ == '__main__':
    loss_l = []
    for i in range(config.epochs):
        model, loss = train(i)
        loss_l.append(loss)
    torch.save(model.state_dict(), "model/seq2seq_model_final.pkl")
    plt.plot(range(config.epochs), loss_l)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    if os.path.exists("model/seq2seq_model_best.pkl"):
        model.load_state_dict(torch.load("model/seq2seq_model_best.pkl"))
    else:
        model.load_state_dict(torch.load("model/seq2seq_model_final.pkl"))
    evaluate(dataset.corpus, './test/', model)
