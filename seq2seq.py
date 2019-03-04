from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import re
import os
import unicodedata
import numpy as np
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from IPython.display import clear_output

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

def load_imdb_data(path, seq_len=40, gen=False):
    """
    Loads IMDB 50k unsupervised reviews
    
    path: str, path to the unsupervised reviews data
    seq_len: minimum length of sequence
    gen: if True all the reviews will be length of seq_len
    """
    reviews = []
    
    for i in tqdm(range(50000)):
        with open(path + f'{i}_0.txt', 'r') as f:
            rev = f.read()
        
        rev = rev.replace(' br ', ' ')
        if len(prog.findall(rev)) >= seq_len:
            if gen:
                reviews.append(['<sos>'] + prog.findall(rev)[:seq_len])
                if len(prog.findall(rev)[:seq_len]) == 39:
                    print(len(rev.split()))
            else:
                reviews.append(['<sos>'] + prog.findall(rev))
    return reviews

def vocab_idxs(data):
    """
    Returns vocab, word2id and id2word, where
    vocab: set of all words in data
    word2id: dictionary that maps words on idxs
    id2word: inverse dictionary to word2id
    
    data: 
    type: list
    format: list of lists of words
    """
    vocab = set()
    for sentence in tqdm(data):
        for s in sentence:
            vocab.add(s)
    word2id = {k:v for v, k in enumerate(vocab, 1)}
    word2id['<m>'] = 0
    id2word = {v:k for k, v in word2id.items()}
    return vocab, word2id, id2word

def sents2matrix(data, word2id, seq_len=41):
    """
    Returns a matrix of integers
    where each row represents a sentence
    
    data:
    type: list
    format: list of lists of words of the seq_len length
    example: [['hello', 'world'], ['nice', 'day']]
    ----------------------------------------------------
    
    word2id: dict that maps word on idxs
    ----------------------------------------------------
    
    seq_len: len of lists contained in data
    """
    
    matrix = np.zeros((len(data), seq_len))
    for i in tqdm(range(len(data))):
        matrix[i] = np.array([int(word2id[word]) for word in data[i]])
    return np.array(matrix)

class MaskedEncoderRNN(nn.Module):
    def __init__(
        self, hidden_dim, vocab_size,
        embedding_dim, p=0.5, n_layers=1, device="cuda"
    ):
        super(MaskedEncoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.device = device
        self.p = p

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, input, hidden):    
        input = input.to(self.device).long()
        mask = self.generate_mask(input.shape).long() # now masked symbols are <m> pad symbol
        masked_input = input * mask
        output = self.embedding(masked_input) #.view(self.n_layers, input.shape[0], -1)
        output, hidden = self.lstm(output, hidden)
        return output, hidden, mask
    
    def generate_mask(self, size):
        return torch.randn(size, device=self.device).ge(self.p)

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device),
            torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        )

class MaskedDecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, embedding_dim, n_layers=1, device="cuda"):
        super(MaskedDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.device = device
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        input = input.long()
        output = self.embedding(input)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(F.relu(self.out(output)))
        
        return output, hidden
    
def plot_history(train_history, title='loss'):
    plt.figure()
    plt.title('{}'.format(title))
    plt.plot(train_history, label='train', zorder=1)    
    plt.xlabel('train steps')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    
def trainIters(encoder, decoder, n_epochs, learning_rate=0.0001, train_on_gpu=True):
    start = time.time()
    train_log = []

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    
    encoder.train()
    decoder.train()

    for epoch in range(n_epochs):
        train_loss = train_epoch(encoder, decoder, encoder_optimizer, decoder_optimizer, train_loader)
        train_log.extend(train_loss)
        
        #clear_output()
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, np.mean(train_log[-100:])))
        #plot_history(train_log)
        
    if save_to_disk:
        torch.save(model, 'generator.pt')
        
def train_epoch(encoder, decoder, encoder_optimizer, decoder_optimizer, train_loader):
    loss_log = []
    criterion = nn.NLLLoss()

    all_input = 50000
    have_now = 0
    for sequence in train_loader:
        input = sequence[0].to(encoder.device)
        output = input
        loss = train(input, output, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        loss_log.append(loss.item())
        have_now += len(train_loader)
        clear_output(True)
        print ('substage [{}/{}], Loss: {:.4f}'.format(have_now, all_input, np.mean(loss_log)))

    return loss_log

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    
    # encoder part
    
    encoder_hidden = encoder.init_hidden(64)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_output, encoder_hidden, mask = encoder(input_tensor, encoder_hidden)
    
    #decoder part
    
    decoder_input = torch.ones(input_tensor.shape[0], 1).to(decoder.device).long()
    decoder_output, decoder_hidden = decoder(decoder_input, encoder_hidden)
    #char_column = input_tensor[:, 0].long()
    
    tmp_output = torch.zeros(decoder_output.shape).to(decoder.device)
    for batch_index in range(input_tensor.shape[0]):
        if mask[batch_index, 0] == 1:
            old_distr = torch.zeros(tmp_output[batch_index, 0].shape).to(decoder.device)
            old_distr[input_tensor[batch_index, 0]] = 1
            tmp_output[batch_index, 0] = old_distr
        else:
            tmp_output[batch_index, 0] = decoder_output[batch_index, 0]
    decoder_output = tmp_output

    loss = criterion(decoder_output.view(input_tensor.shape[0], -1), input_tensor[:, 0])
    
    for char_index in range(input_tensor.shape[1] - 1):
        #decoder_input = input_tensor[:, char_index + 1].to(decoder.device).long().view(-1, 1)
        decoder_output = torch.argmax(decoder_output, dim=2)
        decoder_output, decoder_hidden = decoder(decoder_output, decoder_hidden)
        #char_column = input_tensor[:, char_index + 1]

        tmp_output = torch.zeros(decoder_output.shape).to(decoder.device)
        for batch_index in range(input_tensor.shape[0]):
            if mask[batch_index, char_index + 1] == 1:
                old_distr = torch.zeros(tmp_output[batch_index, 0].shape)
                old_distr[input_tensor[batch_index, char_index + 1]] = 1
                tmp_output[batch_index, 0] = old_distr
            else:
                tmp_output[batch_index, 0] = decoder_output[batch_index, 0]
        decoder_output = tmp_output

        loss += criterion(decoder_output.view(input_tensor.shape[0], -1), input_tensor[:, char_index + 1])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss / input_tensor.shape[1]

if __name__ == "__main__":
    
    prog = re.compile('[A-Za-z0-9]+')
    path = 'aclImdb/train/unsup/'
    
    reviews = load_imdb_data(path, gen=False)
    reviews_40 = load_imdb_data(path, gen=True)
    vocab, word2id, id2word = vocab_idxs(reviews)
    matrix = sents2matrix(reviews_40, word2id)
    
    # create Tensor datasets
    train_data = TensorDataset(torch.LongTensor(matrix))

    # dataloaders
    batch_size = 64
    hidden_dim = 256
    vocab_size = len(vocab)
    embedding_dim = 110
    p = 0.3
    n_layers = 1
    device = torch.device("cuda")

    # make sure the SHUFFLE your training data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    decoder = MaskedDecoderRNN(hidden_dim, vocab_size, embedding_dim, device=device, n_layers=n_layers).to(device)
    encoder = MaskedEncoderRNN(hidden_dim, vocab_size, embedding_dim, device=device, p=p, n_layers=n_layers).to(device)
    
    trainIters(encoder, decoder, n_epochs=30, learning_rate=0.0005)
    
    torch.save(model, "generator.pt")
