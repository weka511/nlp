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

'''
    Continuous Bag Of Words as described in 
    Efficient Estimation of Word Representations in Vector Space
    by Mikolov et al, 2013, https://arxiv.org/abs/1301.3781
'''

__version__ = '1.0'
__author__ = 'Simon Crase'

from argparse import ArgumentParser
from collections.abc import Iterator, Callable

from pathlib import Path
from pickle import dump, HIGHEST_PROTOCOL, load

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, IterableDataset
from torch.nn.functional import one_hot
from matplotlib.pyplot import figure, show
from matplotlib import rc
from matplotlib.ticker import MaxNLocator

from command import Command
from corpus import create_sentence_generator
from vocabulary import Vocabulary
from shared.utils import Logger, user_has_requested_stop, get_seed


class ExampleDataSet(IterableDataset):
    def __init__(self,base,nwords=-1,maxseq=-1,data='./data'):
        self.base = base
        self.data = data
        self.maxseq = maxseq
        self.file_seq = -1
        self.seq = 1
        self.length = -1
        self.words = np.zeros((0),dtype=int)
        self.contexts = np.zeros((0,4),dtype=int)
        self.nwords = nwords
        
    def generate(self):
        while True:
            if self.seq > self.length:
                self.file_seq += 1
                if self.file_seq > self.maxseq: return
                file_name = (Path(self.data) / f'{self.base}-{self.file_seq:03d}').with_suffix('.pkl')
                self.load(file_name)
                Logger.get_instance().log(f'{__file__} {Logger.get_line()} Loaded {file_name}')
                self.seq = 0
                
            yield (self.contexts[self.seq, :], self.get_one_hot(self.words[self.seq]))
            self.seq += 1
            if self.seq%10000 == 0:
                Logger.get_instance().log(f'{__file__} {Logger.get_line()}  {self.seq} {self.length}')
    
    def get_one_hot(self,word):
        return torch.Tensor.float(
                                one_hot(
                                    torch.tensor(word),
                                    num_classes=self.nwords))
        
    def load(self,file_name):
        with open(file_name, 'rb') as inp:
            params = load(inp)
            self.words = params['words']
            self.contexts = params['contexts']
            self.length = len(self.words)
        
    def __iter__(self):
        return iter(self.generate())   


class Examples:
    '''
    This class represents a set of training examples for Continuous Bag of Words,
    
    Attributes:
        window_size Half size of window (context extends left and right)
        vocabulary  Mapping between words and tokens
        contexts    Array of contexts (left and right) for each word
        words       Array of words that occue in each context
    '''
    @staticmethod
    def create(file_name: str) -> 'Examples':
        '''
        A factory method to instantiate a set of Examples from a saved file
        
        Parameters:
            file_name    Name of file where examples have been stored
        '''      
        with open(file_name, 'rb') as inp:
            params = load(inp)
            product = Examples(params['window_size'],
                               vocabulary=params['vocabulary'],
                               maxseq=params['maxseq'])
            Logger.get_instance().log(f'{__file__} {Logger.get_line()} Loaded examples from {file_name.resolve()}')
            return product

    def __init__(self,
                 window_size: int = 4,
                 vocabulary = Vocabulary(sentence_tokens=True),
                 maxseq = -1):
        '''
        Parameters:
            window_size      Half size of window (context extends left and right)
        '''
        self.window_size = window_size
        self.vocabulary = vocabulary
        self.contexts = np.full((0, 2 * self.window_size), -1, dtype=int)
        self.words = np.full((0), -1, dtype=int)
        self.maxseq = maxseq

    def build(self, sentences: Iterator[str],
              rng=np.random.default_rng(),
              max_sentences=None,
              freq=1000,
              segment_size=100000):
        '''
        Construct tables of words and contextx
        
        Parameters:
            sentences       A generator that returns a sentence at a time
            rng             Random number generator
            max_sentences   Maximum number of sentences to be processed
            freq            Controls reporting: report every freq setences
        '''
        words = []
        contexts = []
        n_training_examples = 0
        
        for i,sentence in enumerate(sentences):
            if max_sentences != None and i > max_sentences: break
            try:
                w, c = self.__accumulate__(self.__tokenize__(sentence))
                words.append(w)
                contexts.append(c)                    
            except ValueError:      # Some sentences are too short: ignore them
                pass
            
            if i > 0 and i%freq == 0:
                Logger.get_instance().log(f'{__file__} {Logger.get_line()} Processed {i} sentences')
                
            if len(words) > segment_size:
                n_training_examples += len(words)
                yield np.concatenate(words), np.concatenate(contexts)
                words = []
                contexts = []
                
        if len(words) > 0:
            n_training_examples += len(words)
            yield np.concatenate(words), np.concatenate(contexts)
    
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Created {n_training_examples} training examples')
            

    def create_datasets(self):
        '''
        Used to extract training and test datasets
        '''
        return (ExampleDataSet(self.contexts,self.words,self.vocabulary),
                ExampleDataSet(self.contexts_test,self.words_test,self.vocabulary))
        
    def __tokenize__(self, sentence: Iterator[str]) -> [int]:
        '''
        Convert a sentence to tokens
        
        Parameters:
            sentence    A list of words from one sentence
            
        Returns:
           List of tokens
        '''
        return (
            [self.vocabulary.SOS] +
            [self.vocabulary.tokenize(word) for word in sentence] +
            [self.vocabulary.EOS]
        )

    def __accumulate__(self, tokens: [int]):
        '''
        Convert a sequence of tokens, representing one sentence, to 
        words and contexts
        
        Parameters:
            tokens
        '''
        start = 0
        end = start + 2 * self.window_size + 1
        n_entries = len(tokens) - end + 1
        context = np.full((n_entries, 2 * self.window_size), -1, dtype=int)
        words = np.full((n_entries), -1, dtype=int)
        while end <= len(tokens):
            run = [tokens[i] for i in range(start, end)]
            words[start] = run[self.window_size]
            context[start, :self.window_size] = run[:self.window_size]
            context[start, self.window_size:] = run[self.window_size + 1:]
            start += 1
            end += 1
        return words, context

    def save(self, output:str, maxseq=-1, data: str = './data'):
        '''
        Save Examples using pickle.
        
        Parameters:
            file     Name of file where tables will be saved
            report
        '''
        file = (Path(data) / output).with_suffix('.pkl')
        with open(file, 'wb') as out:
            dump({
                'window_size' : self.window_size,
                'vocabulary' : self.vocabulary,
                'maxseq' : maxseq
                }, out, HIGHEST_PROTOCOL)
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Saved examples in {file.resolve()}')

    def save_word_context(self,seq,words,contexts,output:str, data: str = './data'):
        file = (Path(data) / f'{output}-{seq:03d}').with_suffix('.pkl')
        with open(file, 'wb') as out:
            dump({
                'words' : words,
                'contexts' : contexts
            }, out, HIGHEST_PROTOCOL)
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Saved words and contexts in {file.resolve()}')        

class WordEmbeddings(nn.Module):
    '''
    Thic class represents the CBOW Network from Mikolov et al 2013
    '''

    def __init__(self, V, D, n):
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
        examples = Examples(window_size=args.window_size)
 
        for seq,(words,contexts) in enumerate(examples.build(
                                create_sentence_generator(args.input,args.data),
                                rng=rng,
                                max_sentences=args.max_sentences)):
            examples.save_word_context(seq,words,contexts,args.output,data=args.data)

        examples.save(args.output,maxseq=seq,data=args.data)


class TrainWordEmbeddings(Command):
    '''
    Command to train model
    '''

    def __init__(self):
        super().__init__('train')
        
    def seed_hook(self,seed):
        '''
        Pass seed into torch
        '''
        super().seed_hook(seed)
        torch.manual_seed(seed)

    def _execute(self, args, rng=np.random.default_rng()):
        '''
        Train model and plot losses
        '''
        examples = Examples.create((Path(args.data) / args.input[0]).with_suffix('.pkl'))

        model = WordEmbeddings(len(examples.vocabulary), args.embedding_dim, args.window_size)
        if args.reload != None:
            load_file = (Path(args.data) / args.reload).with_suffix('.pth')
            model.load_state_dict(torch.load(load_file, weights_only=True))
            Logger.get_instance().log(f'{__file__} {Logger.get_line()} Reloaded weights from {load_file}')
            model.eval()
            
        train_loss = train(model, 
                           dataloader=DataLoader(
                                            ExampleDataSet(args.input[0],
                                                           nwords=len(examples.vocabulary),
                                                           maxseq=examples.maxseq,
                                                           data=args.data),
                                            batch_size=args.batch),
                           NIterations=args.NIterations
                        )
        save_file = (Path(args.data) / args.output).with_suffix('.pth')
        torch.save(model.state_dict(), save_file)
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Saved weights to {save_file}')
        self._plot_losses(train_loss,  args.figs, args.output, args.NIterations,
                          title=f'{args.input[0]}: half window size={examples.window_size},'
                          f' embedding dimension={args.embedding_dim}, lr={args.lr}, momentum={args.momentum}')

    def _plot_losses(self, training_losses: [float],  figs: str, output: str, NIterations: int,
                     title=None):
        '''
        Plot training and test losses
        
        Parameters:
            training_losses
            test_losses
            figs
            output
            NIterations
        '''
        fig = figure(figsize=(12, 12))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(list(range(1, len(training_losses) + 1)), training_losses, label=f'Training {training_losses[-1]:.6}', c='xkcd:blue')
        #ax1.plot(list(range(1, len(training_losses) + 1)), test_losses, label=f'Test {test_losses[-1]:.6}', c='xkcd:red')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        _, ymax = ax1.get_ylim()
        ax1.set_ylim(0, ymax)
        #ax1.vlines(best_epoch,0,ymax,colors='xkcd:red',linestyles='dashed',label='Best')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(title='Losses')
        if title != None:
            ax1.set_title(title)
            
        fig.savefig((Path(figs) / output).with_suffix('.png'))
        
def parse_args(choices: [str]):
    '''
    Parse command line arguments
    '''
    data = './data'
    logs = './logs'
    figs = './figs'
    window_size = 4
    batch = 64
    test_set_size = 0.1
    lr = 0.01
    momentum = 0.95
    
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=choices, help='Selects the function that is to be executed')
    parser.add_argument('input', nargs='+', help='List of input files')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generation')
    parser.add_argument('--data', default=data, help=f'Path to data files [{data}]')
    parser.add_argument('--logs', default=logs, help=f'Location for storing log files [{logs}]')
    parser.add_argument('-o', '--output', default=None, required=True, help='File name for storing results')
    
    group_create_examples = parser.add_argument_group('examples','Options for creating training examples')
    group_create_examples.add_argument('-m', '--window_size', type=int, default=window_size, 
                                    help=f'Half size of window (context extends left and right) [{window_size}]')
    group_create_examples.add_argument('-n', '--max_sentences', type=int, default=None, 
                                    help=f'Maximum number of sentences')    
    
    group_train_embeddings = parser.add_argument_group('train','Options for Training')
    group_train_embeddings.add_argument('-D', '--embedding_dim', type=int, default=300,help='Dimensionality of Embedding vectors')
    group_train_embeddings.add_argument('-N', '--NIterations', type=int, default=100,
                                        help='Number of epochs for training')
    group_train_embeddings.add_argument('--lr',type=float,default=lr,help=f'Learning rate [{lr}]')
    group_train_embeddings.add_argument('--momentum',type=float,default=momentum,help=f'Momentum = [{momentum}]')
    group_train_embeddings.add_argument('--batch', type=int, default=batch,help='Batch size for training')
    group_train_embeddings.add_argument('--reload', default=None,
                                        help='Indicates that weights are to be reloaded from file before training')
    group_train_embeddings.add_argument('--show', default=False, action='store_true', help='Controls whether plots are shown')
    group_train_embeddings.add_argument('--figs', default=figs, help=f'Path used to store plots [{figs}]')    
    
    return parser.parse_args()


def train(
        model: 'WordEmbeddings',
        dataloader: 'DataLoader' = None,
        NIterations: int = 100,
        lr:float=0.01, 
        momentum:float=0.95
    ) -> [float]:
    '''
    Train model and compute training losses
    
    Parameters:
        model
        train_dataloader
        test_dataloader
        NIterations
        
    Returns:
       Training losses for each epoch
       Test Losses for each epoch
    '''
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    running_train_loss = []
  
    for epoch in range(NIterations):       
        train_loss = []
        model.train()

        for feature, label in dataloader:
            y_train_pred = model(feature.to(device))
            loss = loss_fn(y_train_pred, label.to(device))
            train_loss.append(loss.item() * feature.size(0)) # Scale by batch size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_train_loss = np.sum(train_loss) #/ len(dataloader)
        running_train_loss.append(mean_train_loss)

        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Epoch: {epoch}, Mean Training Loss = {mean_train_loss}')
            
        if user_has_requested_stop():
            Logger.get_instance().log(f'{__file__} {Logger.get_line()} Stopping')
            break 
 
    return running_train_loss


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


if __name__ == '__main__':
    main()
