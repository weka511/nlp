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

'''Continuous Bag Of Words'''

__version__ = '0.0'
__author__ = 'Simon Crase'

from argparse import ArgumentParser
from glob import glob
from os.path import join
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import one_hot
from matplotlib.pyplot import figure,show

from vocabulary import Vocabulary
from tokenizer import generate_sentences, generate_text, generate_tokens, Token
from shared.utils import Logger, user_has_requested_stop, get_seed

class Examples:
    def __init__(self,window=4):
        self.window = window
        self.vocabulary = Vocabulary(sentence_tokens=True)
        context = np.full((0,2*self.window),-1,dtype=int)
        words = np.full((0),-1,dtype=int)
     
    def __len__(self):
        m, = self.words.shape
        return m
    
    def __getitem__(self, idx):
        label = torch.Tensor.float(one_hot(torch.tensor(self.words[idx]),num_classes=len(self.vocabulary)))
        return self.contexts[idx,:],label
        
    def build(self, sentence_generator):
        words = []
        contexts = []

        for sentence in sentence_generator:
            try:
                w,c = self.accumulate(self.tokenize (sentence))
                words.append(w)
                contexts.append(c)
            except ValueError:      # Some sentences are too short: ignore them
                pass
                
        self.words = np.concatenate(words)
        self.contexts = np.concatenate(contexts)
  
              
    def tokenize(self,sentence):
        return (
            [self.vocabulary.SOS] +
            [self.vocabulary.tokenize(word) for word in sentence] +
            [self.vocabulary.EOS]
        )
    
    def accumulate(self,tokens):
        start = 0
        end = start + 2*self.window + 1
        n_entries = len(tokens) - end +1
        context = np.full((n_entries,2*self.window),-1,dtype=int)
        words = np.full((n_entries),-1,dtype=int)
        while end <= len(tokens):
            run = [tokens[i] for i in range(start,end)]
            words[start] = run[self.window]
            context[start,:self.window] = run[:self.window]
            context[start,self.window:] = run[self.window+1:]
            start += 1
            end += 1
        return words,context
    
        

class WordEmbeddings(nn.Module):
    '''
    Thic class represengts the CBOW Network frm Mikolov et al 2013
    '''
    def __init__(self,V,D,n):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=V*n, embedding_dim=D)
        self.linear_1 = nn.Linear(in_features=D, out_features=V)

    def forward(self, x):
        x = self.embeddings(x)
        x = x.mean(axis=1)
        return self.linear_1(x)
    
def parse_args():
    data = './data'
    logs = './logs'
    figs = './figs'
    window = 4
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='+', help='List of input files')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generation')
    parser.add_argument('--data', default=data, help=f'Path to data files [{data}]')    
    parser.add_argument('-D','--embedding_dim',type=int,default=300)
    parser.add_argument('-n','--windows_size',type=int,default=4)
    parser.add_argument('-N','--NIterators',type=int,default=100)
    parser.add_argument('--batch',type=int, default=4)
    parser.add_argument('--logs', default=logs, help=f'Location for storing log files [{logs}]')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plots are shown')
    parser.add_argument('--figs', default=figs, help=f'Path used to store plots [{figs}]')
    parser.add_argument('-w', '--window', type=int, default=window, help=f'Width of window for context [{window}]')
    parser.add_argument('-o', '--output', default=None, required=True, help='File name for storing results')
    
    return parser.parse_args()

def train(model,dataloader,NIterators=100):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.95)
    loss_fn = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    running_loss=[]
    for epoch in range(NIterators):
        train_loss = []
        model.train()
        
        for feature, label in dataloader:
            model = model.to(device)
            y_train_pred = model(feature.to(device))
            loss = loss_fn(y_train_pred, label.to(device))
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_train_loss = np.average(train_loss)
        print(f'Epoch:{epoch} | Mean Training Loss : {mean_train_loss}') 
        running_loss.append(mean_train_loss)
    return running_loss
        
def main():
    start  = time()
    args = parse_args()
    with Logger(Path(__file__).stem, path=args.logs) as _:
        seed = get_seed(args.seed,
                        notify=lambda s: Logger.get_instance().log(f'{__file__} {Logger.get_line()}'
                                                                   f' Created new seed {s}'))    
        torch.manual_seed(seed)
        examples = Examples(window=args.window)
    
        examples.build(
                generate_sentences(
                    generate_tokens(
                        generate_text(
                            file_names=[globbed for name in args.input for globbed in glob(join(args.data, name))]
                        ))))
        m = len(examples)
        
        model = WordEmbeddings(len(examples.vocabulary),args.embedding_dim,args.windows_size)
     
        train_dataloader = DataLoader(examples, batch_size=args.batch, shuffle=True)
        training_losses = train(model,train_dataloader,NIterators=args.NIterators)
        
        fig = figure(figsize=(12,12))
        ax1 = fig.add_subplot(1,1,1)
        ax1.plot(training_losses,label=f'Training Losses {training_losses[-1]:.6}')
        ax1.legend()
        fig.savefig((Path(args.figs) / args.output).with_suffix('.png'))
        
        elapsed = time() - start
        minutes = int(elapsed/60)
        seconds = elapsed - 60*minutes
        print (f'Elapsed Time {minutes} m {seconds:.2f} s')
        if args.show:
            show()
    
if __name__=='__main__':
    main()
