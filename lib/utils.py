import re
from tqdm import tqdm
import numpy as np


def load_imdb_data(path, seq_len=40, gen=False):
    """
    Loads IMDB 50k unsupervised reviews
    
    path: str, path to the unsupervised reviews data
    seq_len: minimum length of sequence
    gen: if True all the reviews will be length of seq_len
    """
    prog = re.compile('[A-Za-z0-9]+')
    
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
    vocab.remove("<sos>")
    word2id = {k:v for v, k in enumerate(vocab, 2)}
    word2id['<m>'] = 0
    word2id['<sos>'] = 1
    vocab.add("<sos>")
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
