import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np



class DiscriminatorEncoder(nn.Module):
    
    def __init__(self, hidden_dim, vocab_size, embedding_dim, 
                 n_layers=1, train_on_gpu=True, bidirectional=False,
                 p=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.train_on_gpu = train_on_gpu
        self.bidirectional = bidirectional
        self.p = p
        
        # Embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim=embedding_dim)
        
        # LSTM
        dropout = 1 if n_layers > 1 else 0
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            batch_first=True, dropout=dropout, bidirectional=bidirectional)
        
        if bidirectional:
            self.projection = nn.Linear(hidden_dim*2, hidden_dim)
      
    
    def forward(self, x, h):
        ''' Forward pass through the network. 
            x are inputs and the hidden/cell state `hidden`. '''
        
        # now masked symbols are <m> pad symbol
        mask = self.generate_mask(x.shape) 
        masked_input = torch.mul(x, mask)
        
        x = self.embeddings(x)
        output, hidden = self.lstm(x, h)
        if self.bidirectional:
            output = self.projection(output)
            hidden = (self.projection(torch.cat((hidden[0][0], hidden[0][1]), 1).unsqueeze(0)), 
                      self.projection(torch.cat((hidden[1][0], hidden[1][1]), 1).unsqueeze(0)))
        
        return output, hidden, mask
    
    def generate_mask(self, size):
        mask = np.random.choice(2, size, p=[1-self.p, self.p])
        if self.train_on_gpu:
            return torch.from_numpy(mask).long().cuda()
        else:
            return torch.from_numpy(mask).long()

    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (self.train_on_gpu):
            hidden = (weight.new(self.n_layers*(self.bidirectional + 1), batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers*(self.bidirectional + 1), batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers*(self.bidirectional + 1), batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers*(self.bidirectional + 1), batch_size, self.hidden_dim).zero_())
        
        return hidden

# ------------------------------------------------------------------------------------------------------------------------    

class DiscriminatorDecoder(nn.Module):
    def __init__(self, hidden_size, embedding_dim, dropout_p=0.1, 
                 n_layers=1, max_length=41, train_on_gpu=True):
        super(AttnMaskedDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        
        # embeddings
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        
        # attention layer
        self.attention = Attention(hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        
        # lstm layer
        self.lstm = nn.LSTM(embedding_dim, self.hidden_size, n_layers, batch_first=True)
        
        # final fully-connected layer
        self.prediction = nn.Linear(2*hidden_size, 2)

    def forward(self, input, hidden, encoder_outputs):
        # embeddings
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        
        # attention
        context, attn_weights = self.attention(hidden[0].view(input.size(0), 1, -1), encoder_outputs)
        
        # lstm
        output, hidden = self.lstm(embedded, hidden)
        
        # lstm outs + context vector
        output = torch.cat((output, context), dim=2)
        
        # result
        output = F.log_softmax(self.prediction(output), dim=2)
        
        return output, hidden, attn_weights

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (self.train_on_gpu):
            hidden = (weight.new(self.n_layers*(self.bidirectional + 1), batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers*(self.bidirectional + 1), batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers*(self.bidirectional + 1), batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers*(self.bidirectional + 1), batch_size, self.hidden_dim).zero_())
        
        return hidden


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.view(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.view(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights
