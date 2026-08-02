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

__version__ = '0.1'
__author__ = 'Simon Crase'

from argparse import ArgumentParser
from glob import glob
from os.path import join
from pathlib import Path
from pickle import dump, HIGHEST_PROTOCOL, load

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import one_hot
from matplotlib.pyplot import figure, show
from matplotlib import rc
from matplotlib.ticker import MaxNLocator

from command import Command
from vocabulary import Vocabulary
from tokenizer import generate_sentences, generate_text, generate_tokens, Token
from shared.utils import Logger, user_has_requested_stop, get_seed

class Examples:
    '''
    This class represents a set of training examples for CBOW
    '''
    @staticmethod
    def create(file_name:str) -> 'Examples':
        '''
        A factory method to instantiate a set of Examples from a saved file
        
        Parameters:
            file_name    Name of file where examples have been stored
        '''
        with open(file_name, 'rb') as inp:
            product = load(inp)
            Logger.get_instance().log(f'{__file__} {Logger.get_line()} Loaded examples from {file_name.resolve()}')
            return product
        
    def __init__(self,window : int=4):
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
    
    def save(self, file, report=lambda s: Logger.get_instance().log(f'{__file__} {Logger.get_line()} {s}')):
        '''
        Save Examples using pickle.
        
        Parameters:
            file     Name of file where tables will be saved
        '''
        with open(file, 'wb') as out:
            dump(self, out, HIGHEST_PROTOCOL)
            report(f'Saved examples in {file.resolve()}')        

class WordEmbeddings(nn.Module):
    '''
    Thic class represengts the CBOW Network from Mikolov et al 2013
    '''
    def __init__(self,V,D,n):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=V*n, embedding_dim=D)
        self.linear_1 = nn.Linear(in_features=D, out_features=V)

    def forward(self, x):
        x = self.embeddings(x)
        x = x.mean(axis=1)
        return self.linear_1(x)

    
class CreateExamples(Command):
    '''
    Build examples for training CBOW
    '''
    def __init__(self):
        super().__init__('examples')
        
    def _execute(self, args, rng=np.random.default_rng()):
        '''
        Parse text into tokens, then build examples
        
        Parameters:
            args       Command line parameters as parsed by parse_args()
            rng        Random number generator
        '''
        examples = Examples(window=args.window)
    
        examples.build(
                generate_sentences(
                    generate_tokens(
                        generate_text(
                            file_names=[globbed for name in args.input for globbed in glob(join(args.data, name))]
                        ))))  
        
        examples.save((Path(args.data) / args.output).with_suffix('.pkl'),
                      report=lambda s: Logger.get_instance().log(f'{__file__} {Logger.get_line()} {s}'))    

class TrainWordEmbeddings(Command):
    def __init__(self):
        super().__init__('train')
        
    def _execute(self, args, rng=np.random.default_rng()):
        examples = Examples.create((Path(args.data) / args.input[0]).with_suffix('.pkl'))
        model = WordEmbeddings(len(examples.vocabulary),args.embedding_dim,args.windows_size)
        if args.reload != None:
            load_file = (Path(args.data) / args.reload).with_suffix('.pth')
            model.load_state_dict(torch.load(load_file, weights_only=True))
            Logger.get_instance().log(f'{__file__} {Logger.get_line()} Reloaded weights from {load_file}')
            model.eval()    
        train_dataloader = DataLoader(examples, batch_size=args.batch, shuffle=True)
        training_losses = train(model,train_dataloader,NIterations=args.NIterations)
        save_file = (Path(args.data) / args.output).with_suffix('.pth')
        torch.save(model.state_dict(), save_file)
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Saved weights to {save_file}')
        self._plot_losses(training_losses,args.figs,args.output,args.NIterations)
        
    def _plot_losses(self,training_losses,figs,output,NIterations):
        fig = figure(figsize=(12,12))
        ax1 = fig.add_subplot(1,1,1)
        ax1.plot(list(range(1,NIterations+1)),training_losses,label=f'Training Losses {training_losses[-1]:.6}')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        _,ymax = ax1.get_ylim()
        ax1.set_ylim(0,ymax)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        fig.savefig((Path(figs) / output).with_suffix('.png'))   
    
def parse_args(choices : [str]):
    data = './data'
    logs = './logs'
    figs = './figs'
    window = 4
    batch = 64
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=choices, help='Selects the function that is to be executed')
    parser.add_argument('input', nargs='+', help='List of input files')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generation')
    parser.add_argument('--data', default=data, help=f'Path to data files [{data}]')    
    parser.add_argument('-D','--embedding_dim',type=int,default=300)
    parser.add_argument('-n','--windows_size',type=int,default=4)
    parser.add_argument('-N','--NIterations',type=int,default=100)
    parser.add_argument('--batch',type=int, default=batch)
    parser.add_argument('--logs', default=logs, help=f'Location for storing log files [{logs}]')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plots are shown')
    parser.add_argument('--figs', default=figs, help=f'Path used to store plots [{figs}]')
    parser.add_argument('-w', '--window', type=int, default=window, help=f'Width of window for context [{window}]')
    parser.add_argument('-o', '--output', default=None, required=True, help='File name for storing results')
    parser.add_argument('--reload', default=None)  
    
    return parser.parse_args()

def train(model,dataloader,NIterations=100):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.95)
    loss_fn = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    running_loss=[]
    for epoch in range(NIterations):
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
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)

    Command.append([
        CreateExamples(),
        TrainWordEmbeddings()
    ])
    
    args = parse_args(Command.get_choices())
    Command.get_command(args.command).execute(args)

    if args.show:
        show()    
    
if __name__=='__main__':
    main()
