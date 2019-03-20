import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

class LanguageModel(nn.Module):
    
    def __init__(self, hidden_dim, vocab_size, embedding_dim, 
                 linear_dim=128, n_layers=3, train_on_gpu=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.train_on_gpu = train_on_gpu
        
        # Embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim=embedding_dim)
        
        # LSTM
        dropout = 1 if n_layers > 1 else 0
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        
        # fully-connected layes
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, linear_dim),
            nn.ReLU(),
            nn.Linear(linear_dim, linear_dim),
            nn.ReLU(),
            nn.Linear(linear_dim, vocab_size)
        )
      
    
    def forward(self, x, h):
        ''' Forward pass through the network. 
            x are inputs and the hidden/cell state `hidden`. '''
        
        x = self.embeddings(x)
        
        output, hidden = self.lstm(x, h)
        output = output.contiguous().view(-1, self.hidden_dim)
        output = self.fc(output)
        output = F.log_softmax(output, dim=1)
        
        return output, hidden

    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (self.train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

# --------------------------------------------------------------------------------------------
# train 
    
def eval_epoch(model, eval_loader, eval_on_gpu=True):
    criterion = nn.NLLLoss()
    loss_log = []
    model.eval()
    for sequence in eval_loader:
        #init hidden
        h = model.init_hidden(sequence[0].size(0))
        h = tuple([each.data for each in h])
        # switch to gpu/cpu
        if eval_on_gpu:
            X = sequence[0][:, :-1].cuda()
            y = sequence[0][:, 1:].cuda()
        else:
            X = sequence[0][:, :-1]
            y = sequence[0][:, 1:]
        
        output, hidden = model(X, h)
        loss = criterion(output, y.contiguous().view(-1))
        loss_log.append(loss.item())
    return loss_log

def train_epoch(model, optimizer, train_loader, train_on_gpu=True):
    criterion = nn.NLLLoss()
    loss_log = []
    model.train()
    for sequence in train_loader:
        optimizer.zero_grad()
        h = model.init_hidden(sequence[0].size(0))
        h = tuple([each.data for each in h])
        if train_on_gpu:
            X = sequence[0][:, :-1].cuda()
            y = sequence[0][:, 1:].cuda()
        else:
            X = sequence[0][:, :-1]
            y = sequence[0][:, 1:]
        output, hidden = model(X, h)
        loss = criterion(output, y.contiguous().view(-1))
        loss.backward()
        optimizer.step()
        loss_log.append(loss.item())
    return loss_log   

def plot_history(train_history, title='loss'):
    plt.figure()
    plt.title('{}'.format(title))
    plt.plot(train_history, label='train', zorder=1)    
    plt.xlabel('train steps')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    
def train(model, opt, n_epochs, train_loader, train_on_gpu=True, save_to_disk=True, path='pretrained_model.pt'):
    train_log = []
    total_steps = 0
    
    if train_on_gpu:
        model.cuda()
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, opt, train_loader, train_on_gpu=train_on_gpu)
        train_log.extend(train_loss)
        total_steps += len(train_loader)
        
        clear_output()
        print ('Epoch [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, n_epochs, np.mean(train_log[-100:])))
        plot_history(train_log)
        
        if epoch % 5 == 0:
            if save_to_disk:
                torch.save(model, path)
        
def eval_model(model, eval_loader, eval_on_gpu=True):
    eval_log = []
    
    if eval_on_gpu:
        model.cuda()
    eval_loss = eval_epoch(model, eval_loader)
    eval_log.extend(eval_loss)

    clear_output()
    plot_history(eval_log)
    return eval_log
