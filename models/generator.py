import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedEncoderRNN(nn.Module):
    def __init__(
        self, hidden_dim, vocab_size,
        embedding_dim, p=0.5, n_layers=1, device=torch.device("cuda")
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
        input = input.to(self.device)
        mask = self.generate_mask(input.shape) # now masked symbols are <m> pad symbol
        masked_input = torch.mul(input, mask)
        output = self.embedding(input) #.view(self.n_layers, input.shape[0], -1)
        output, hidden = self.lstm(output, hidden)
        return output, hidden, mask
    
    def generate_mask(self, size):
        return torch.randn(size, device=self.device).ge(self.p).long()

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device),
            torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        )


class AttnMaskedDecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, dropout_p=0.1, n_layers=1, max_length=41, device="cuda"):
        super(AttnMaskedDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.device = device
        
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, n_layers, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden[0].transpose(0,1)), 2)), dim=2)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        output = torch.cat((embedded, attn_applied), 2)
        output = self.attn_combine(output)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = F.log_softmax(self.out(output), dim=2)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    

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
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, input, hidden):
        output = self.embedding(input.long())
        output, hidden = self.lstm(output, hidden)
        output = self.fc(output)
        output = F.relu(output)
        output = self.softmax(self.out(output))
        
        return output, hidden
