# Imports 
from __future__ import print_function
from tqdm import tqdm
import numpy as np
import chainer
import os
import re
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------------------------------


def vocab_idxs(data, sos_token=False):
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
    for sentence in data:
        for s in sentence:
            vocab.add(s)
    if sos_token:
        vocab.remove("<sos>")
        word2id = {k:v for v, k in enumerate(vocab, 2)}
        word2id['<sos>'] = 1
    else:
        word2id = {k:v for v, k in enumerate(vocab, 1)}
        word2id['<m>'] = 0
    vocab.add("<sos>")
    id2word = {v:k for k, v in word2id.items()}
    return vocab, word2id, id2word


def sents2matrix(data, word2id):
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
    
    matrix = np.zeros((len(data), len(data[0])))
    for i in range(len(data)):
        matrix[i] = np.array([int(word2id[word]) for word in data[i]])
    return np.array(matrix)


# -----------------------------------------------------
# IMDB

def prepare_imdb_data(batch_size=64, validation_size=0.15, path='', 
                      seq_len=40, gen=True, return_data=False):
    """
    Prepares imdb data for further work with pytorch
    
    returns: train_loader, valid_loader, 
             vocab, word2id, id2word 
             and original reviews if return_data=True
    -------------------------------------------------
    
    validation_size: size of valdation set
    -------------------------------------------------
    
    for information about other parameters check the 
    doc of the load_imdb_data function
    -------------------------------------------------
    """
    
    reviews = load_imdb_data(path='', seq_len=40, gen=True)
    vocab, word2id, id2word = vocab_idxs(reviews)
    matrix = sents2matrix(reviews, word2id)
    
    # Splitting data into train and validation
    idx = np.random.choice(range(len(matrix)), 
                           size=len(matrix), replace=False)
    split = int(len(idx) * (1 - validation_size))
    train_idx = idx[:split]
    valid_idx = idx[split:]
    
    train_data = TensorDataset(torch.LongTensor(matrix[train_idx]))
    valid_data = TensorDataset(torch.LongTensor(matrix[valid_idx]))
    
    # Loading data to DataLoaders
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    print('Data has been successfully loaded')
    if return_data:
        return train_loader, valid_loader, vocab, word2id, id2word, reviews
    else:
        return train_loader, valid_loader, vocab, word2id, id2word

    
def load_imdb_data(path='', seq_len=40, gen=True):
    """
    Loads IMDB 100k reviews
    
    path: str, path to the aclImdb folder
    seq_len: minimum length of sequence
    gen: if True all the reviews will be length of seq_len
         otherwise function will return sequences of len >= seq_len
    """
    prog = re.compile('[A-Za-z0-9]+')
    
    paths = [path+'aclImdb/test/pos', path+'aclImdb/test/neg', 
             path+'aclImdb/train/pos', path+'aclImdb/train/neg', 
             path+'aclImdb/train/unsup/']
    
    
    reviews = []
    for p in paths:
        files = os.listdir(p)
        for file in files:
            with open(p + '/' + file, 'r') as f:
                rev = f.read()

            rev = rev.replace(' br ', ' ')
            if len(prog.findall(rev)) >= seq_len:
                if gen:
                    reviews.append(['<sos>'] + prog.findall(rev)[:seq_len])
                else:
                    reviews.append(['<sos>'] + prog.findall(rev))
    return reviews

# -----------------------------------------------------
# PTB

def preprocess_ptb(train, ptb_dict, seq_len=40):
    ptb_word_id_dict = ptb_dict
    ptb_id_word_dict = dict((v,k) for k,v in ptb_word_id_dict.items())
    data = []
    for i in tqdm(range(0, len(train) - seq_len, seq_len)):
        data.append([ptb_id_word_dict[i] for i in train[i:i+seq_len]])
    return data


def prepare_ptb_data(batch_size=64, validation_size=0.15, path='', 
                      seq_len=40, gen=True, return_data=False):
    """
    Prepares imdb data for further work with pytorch
    
    returns: train_loader, valid_loader, 
             vocab, word2id, id2word 
             and original reviews if return_data=True
    -------------------------------------------------
    
    validation_size: size of valdation set
    -------------------------------------------------
    
    for information about other parameters check the 
    doc of the load_imdb_data function
    -------------------------------------------------
    """
    train, _, _ = chainer.datasets.get_ptb_words()
    ptb_dict = chainer.datasets.get_ptb_words_vocabulary()
    data = preprocess_ptb(train, ptb_dict, seq_len=seq_len)
    
    vocab, word2id, id2word = vocab_idxs(data)
    matrix = sents2matrix(data, word2id)
    
    # Splitting data into train and validation
    idx = np.random.choice(range(len(matrix)), 
                           size=len(matrix), replace=False)
    split = int(len(idx) * (1 - validation_size))
    train_idx = idx[:split]
    valid_idx = idx[split:]
    
    train_data = TensorDataset(torch.LongTensor(matrix[train_idx]))
    valid_data = TensorDataset(torch.LongTensor(matrix[valid_idx]))
    
    # Loading data to DataLoaders
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    print('Data has been successfully loaded')
    if return_data:
        return train_loader, valid_loader, vocab, word2id, id2word, reviews
    else:
        return train_loader, valid_loader, vocab, word2id, id2word
