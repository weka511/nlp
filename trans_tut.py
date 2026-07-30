#!/usr/bin/env python

#   Copyright (C) 2026 Simon Crase

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 # https://youtu.be/Ub3GoFaUcds?t=2253
'''
    Code snarfed from 
    Transformer Model Tutorial in PyTorch: From Theory to Code, by Arjun Aarkar
    https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
    I have also followed  Chapter 9 of 	
    Speech and Language Processing (3rd ed. draft)
    Dan Jurafsky and James H. Martin .
'''

from argparse import ArgumentParser
from pathlib import Path
from time import time

import numpy as np
from matplotlib.pyplot import figure, show
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


class MultiHeadAttention(nn.Module):
    '''
    This class computes the attention between each pair of positions in a sequence. 
    It is able to compare an item of interest to a collection of other items to
    reveal their relevance in the current context.  It consists of  
    multiple attention heads that capture different aspects of the input sequence.
    
    Attributes:
        d_model    Model's dimension
        num_heads  Number of attention heads
        d_k        Dimension of each head's key, query, and value
        W_q        Query transformation weights.
        W_k        Key transformation weights
        W_v        Value transformation weights
        W_o        Output transformation
    '''

    def __init__(self, d_model, num_heads):
        '''
        Attributes:
            d_model    Model's dimension
            num_heads  Number of attention heads
        '''
        super().__init__()
        assert d_model % num_heads == 0, f'd_model [{d_model}] must be divisible by num_heads [{num_heads}]'

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Query, Key, Value, mask=None, fill_value=-1e9):
        '''
        Calculate scaled dot product attention
        
        Parameters:
            Query      Used to consier an input embedding as the current focus of attention.
            Key        Used to consier an input embedding to be a precding input, 
                       and compare it with the current focus of attention.
            Value      Used to ccompute the value of the current focus of attention.
            mask       Used to prevent attention to certain parts, such as padding
            fill_value A large negative number, used for scores that are masked
        '''
        attention_scores = torch.matmul(Query, Key.transpose(-2, -1)) / np.sqrt(self.d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, fill_value)

        attention_probabilities = torch.softmax(attention_scores, dim=-1)

        return torch.matmul(attention_probabilities, Value)

    def split_heads(self, x):
        '''
        Reshape the input to have num_heads for multi-head attention
        
        Parameters:
            x
        '''
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        '''
        Combine the multiple heads back to original shape
        
        Parameters:
            x
        '''
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        '''
        The forward method is where the actual computation happens:
        
        Parameters:
            Q           Query
            K           Key
            V           Values
            mask        Used to prevent attention to certain parts such as padding
        '''
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output transformation
        return self.W_o(self.combine_heads(attn_output))


class PositionWiseFeedForward(nn.Module):
    '''
    This class peovides a feed-forward network that is applied to each position separately
    and identically. It helps in transforming the features learned by the attention mechanisms  
    within the transformer, acting as an additional processing step for the attention outputs.
    
    Attributes:
        fc1
        fc2
        relu
    '''
    def __init__(self, d_model, d_ff):
        '''
        Parameters:
            d_model   Number of dimensions for model
            d_ff      Number of elements in feed forward layer
        '''
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    '''
    This class adds information about the position of tokens within the sequence. 
    Since the transformer model lacks inherent knowledge of the order of tokens, this class helps
    the model to consider the position of tokens in the sequence. The sinusoidal functions used 
    are chosen to allow the model to easily learn to attend to relative positions,
    as they produce a unique and smooth encoding for each position in the sequence.
    '''

    def __init__(self, d_model, max_seq_length):
        '''
        Parameters:
            d_model          Number of encoder and decoder layers
            max_seq_length   Maximum number of tokens in a sentence
        '''
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    '''
    This class defines a single layer of the transformer's encoder. 
    
    Attributes:
        self_attn     Self attention layer
        norm1         First normalization layer
        feed_forward  Feed forward layer
        norm2         Second normalization layer
        dropout       Dropout layer
    '''
    def __init__(self, d_model, num_heads, d_ff, dropout):
        '''
        Initialize Ebcoder
        
        Parameters:
            d_model     Number of dimensions for model
            num_heads   Number of attention heads
            d_ff        Number of elements in feed forward layer
            dropout     probability of an element to be zeroed
        '''
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_output))

class DecoderLayer(nn.Module):
    '''
    This class defines a single layer of the transformer's decoder. 
     
    Attributes:
        self_attn     Self attention layer
        norm1         First normalization layer
        cross_attn    A second attention layer to attend to the encoder's output
        norm2         Second normalization layer
        feed_forward  Feed forward layer
        norm3         Third normalization layer
        dropout       Dropout layer
    '''
    def __init__(self, d_model, num_heads, d_ff, dropout):
        '''
        Parameters:
            d_model     Number of dimensions for model
            num_heads   Number of attention heads
            d_ff        Number of elements in feed forward layer
            dropout     probability of an element to be zeroed
        '''
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm3(x + self.dropout(ff_output))

class Transformer(nn.Module):
    '''
    This class brings together the various components of a Transformer model. It provides 
    an interface for training and inference, encapsulating the complexities of multi-head attention, 
    feed-forward networks, and layer normalization.
    
    Attributes:
        encoder_embedding 
        decoder_embedding 
        positional_encoding
        encoder_layers
        decoder_layers
        fc
        dropout 
    '''
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        '''
        Parameters:
            src_vocab_size   Size of vocabulary for source language
            tgt_vocab_size   Size of vocabulary for target language
            d_model          Number of dimensions for model
            num_heads        Number of heads
            num_layers       Number of encoder and decoder layers
            d_ff             Number of elements in each feed forward layer
            max_seq_length   Maximum number of tokens in a sentence
            dropout          probability of an element to be zeroed
        '''
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        '''
        Parameters:
            src
            tgt
        '''
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        return self.fc(dec_output)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--N', '-N', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--src_vocab_size', type=int, default=5000, help='Size of vocabulary for source language')
    parser.add_argument('--tgt_vocab_size', type=int, default=5000, help='Size of vocabulary for target language')
    parser.add_argument('--d_model', type=int, default=512, help='Number of dimensions for model')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of encoder and decoder layers')
    parser.add_argument('--max_seq_length', type=int, default=100, help='Maximum number of tokens in a sentence')
    parser.add_argument('--d_ff', type=int, default=2048, help='Number of elements in feed forward layer')
    parser.add_argument('--dropout', type=float, default=0.1, help='probability of an element to be zeroed by dropout')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--betas', type=float, default=(0.9, 0.98), nargs=2,
                        help='Coefficients used by Adam optimizer for computing running averages of gradient and its square.')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plots are shown')
    parser.add_argument('--figs', default='./figs', help=f'Path used to store plots')
    parser.add_argument('--output', '-o', default=Path(__file__).name, help=f'File name for output')
    return parser.parse_args()


def train(transformer, N, optimizer, src_data, tgt_data, criterion, tgt_vocab_size):
    '''
    Train transformer
    
    Parameters:
        transformer
        N
        optimizer
        src_data
        tgt_data
        criterion
        tgt_vocab_size
        
    Returns:
        List containing loss at each step
    '''

    transformer.train()

    Losses = []
    for epoch in range(N):
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1])
        loss = criterion(
            output.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1)
        )
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')
        Losses.append(loss.item())
    return Losses


def validate(transformer, src_vocab_size, tgt_vocab_size, max_seq_length, criterion):
    '''
    Validate tansformer against test dataset
    
    Parameters:
        transformer
        src_vocab_size
        tgt_vocab_size
        max_seq_length
        criterion
        
    Returns:
        Validation loss
    '''
    transformer.eval()

    # Generate random sample validation data
    val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

    with torch.no_grad():
        val_output = transformer(val_src_data, val_tgt_data[:, :-1])
        val_loss = criterion(
            val_output.contiguous().view(-1, tgt_vocab_size),
            val_tgt_data[:, 1:].contiguous().view(-1)
        )
        print(f'Validation Loss: {val_loss.item()}')
    return val_loss.item()


def plot_losses(Losses, validation_loss, figs, output):
    '''
    Plot training loss and validation loss
    
    Parameters:
        Losses
        validation_loss
        figs
        output
    '''
    fig = figure(figsize=(12, 12))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(range(len(Losses)), Losses, c='xkcd:blue', label='Training')
    ax1.hlines(validation_loss, 0, len(Losses) - 1,
               colors='xkcd:red',
               label=f'Validation {validation_loss:.04}',
               linestyles='dashed')
    ax1.legend(title='Losses')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    _, y1 = ax1.get_ylim()
    ax1.set_ylim((0, y1))

    fig.savefig((Path(figs) / output).with_suffix('.png'))


def main():
    '''
    Crerate tarnsformer and data, train, validate, and plot losses.
    '''
    start = time()
    args = parse_args()

    transformer = Transformer(args.src_vocab_size, args.tgt_vocab_size, args.d_model, args.num_heads, args.num_layers,
                              args.d_ff, args.max_seq_length, args.dropout)

    # Generate random sample data
    src_data = torch.randint(1, args.src_vocab_size, (64, args.max_seq_length))  # (batch_size, seq_length)
    tgt_data = torch.randint(1, args.tgt_vocab_size, (64, args.max_seq_length))  # (batch_size, seq_length)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=args.lr, betas=args.betas, eps=0.01)

    Losses = train(transformer, args.N, optimizer, src_data, tgt_data, criterion, args.tgt_vocab_size)
    validation_loss = validate(transformer, args.src_vocab_size, args.tgt_vocab_size, args.max_seq_length, criterion)
    plot_losses(Losses, validation_loss, args.figs, args.output)
    
    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()


if __name__ == '__main__':
    main()
