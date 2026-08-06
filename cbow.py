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
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import one_hot
from matplotlib.pyplot import figure, show
from matplotlib import rc
from matplotlib.ticker import MaxNLocator

from command import Command
from corpus import create_sentence_generator
from vocabulary import Vocabulary
from shared.utils import Logger, user_has_requested_stop, get_seed

class ExampleDataSet:
    '''
    This class wraps one set of data, training or test
    
    Attributes:
        contexts
        words
        vocabulary
    '''
    def __init__(self,contexts,words,vocabulary):
        self.contexts = contexts
        self.words = words
        self.vocabulary = vocabulary
        
    def __len__(self):
        '''
        Find number of examples stored
        '''
        m, = self.words.shape
        return m

    def __getitem__(self, idx: int):
        '''
        Retrieve one example 
        
        Parameters:
            idx     Index of example in dataset
            
        Returns:
            context
            word 
        '''
        
        return (self.contexts[idx, :], 
                torch.Tensor.float(
                                one_hot(torch.tensor(self.words[idx]),
                                        num_classes=len(self.vocabulary)))
                )

class Examples:
    '''
    This class represents a set of training examples for Continuous Bag of Words,
    
    Attributes:
        window_size Half size of window (context extends left and right)
        vocabulary  Mapping between words and tokens
        context     Array of contexts (left and right) for eac h word
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
            product = load(inp)
            Logger.get_instance().log(f'{__file__} {Logger.get_line()} Loaded examples from {file_name.resolve()}')
            return product

    def __init__(self, window_size: int = 4):
        '''
        Parameters:
            window_size      Half size of window (context extends left and right)
        '''
        self.window_size = window_size
        self.vocabulary = Vocabulary(sentence_tokens=True)
        self.context_train = np.full((0, 2 * self.window_size), -1, dtype=int)
        self.words_train = np.full((0), -1, dtype=int)
        self.context_test = np.full((0, 2 * self.window_size), -1, dtype=int)
        self.words_test = np.full((0), -1, dtype=int)        

    def build(self, sentences: Iterator[str],
              test_set_size:float = 0.1, 
              rng=np.random.default_rng()):
        '''
        Construct tables of words and contextx
        
        Parameters:
            sentences   A generator that returns a sentence at a time
            test_set_size
        '''
        words_train = []
        contexts_train = []
        words_test = []
        contexts_test = []
        
        for sentence in sentences:
            try:
                w, c = self.__accumulate__(self.__tokenize__(sentence))
                if rng.uniform() < test_set_size:
                    words_test.append(w)
                    contexts_test.append(c)
                else:
                    words_train.append(w)
                    contexts_train.append(c)                    
            except ValueError:      # Some sentences are too short: ignore them
                pass

        self.words_test = np.concatenate(words_test)
        self.contexts_test = np.concatenate(contexts_test)
        self.words_train = np.concatenate(words_train)
        self.contexts_train = np.concatenate(contexts_train)
        m1,_ = self.contexts_train.shape
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Created {m1} training examples')
        m2,_ = self.contexts_test.shape
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Created {m2} test examples')        

    def create_datasets(self):
        '''
        Used to extract training and test datasets
        '''
        return (ExampleDataSet(self.contexts_train,self.words_train,self.vocabulary),
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

    def save(self, file: str, report=lambda s: Logger.get_instance().log(f'{__file__} {Logger.get_line()} {s}')):
        '''
        Save Examples using pickle.
        
        Parameters:
            file     Name of file where tables will be saved
            report
        '''
        with open(file, 'wb') as out:
            dump(self, out, HIGHEST_PROTOCOL)
            report(f'Saved examples in {file.resolve()}')


class WordEmbeddings(nn.Module):
    '''
    Thic class represents the CBOW Network from Mikolov et al 2013
    '''

    def __init__(self, V, D, n):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=V * n, embedding_dim=D)
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
 
        examples.build(
            create_sentence_generator(args.input,args.data),
            test_set_size=args.test_set_size,
            rng=rng
        )

        examples.save((Path(args.data) / args.output).with_suffix('.pkl'),
                      report=lambda s: Logger.get_instance().log(f'{__file__} {Logger.get_line()} {s}'))


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
            
        training_dataset,test_dataset = examples.create_datasets()

        train_loss,test_loss,best_epoch = train(
                                    model, 
                                    train_dataloader=DataLoader(training_dataset,batch_size=args.batch,shuffle=True), 
                                    test_dataloader=DataLoader(test_dataset,batch_size=args.batch,shuffle=True),
                                    NIterations=args.NIterations,
                                    burn=args.burn,
                                    file_for_best=(Path(args.data) / (args.output.split('.')[0]+'_best')).with_suffix('.pth')
                                )
        save_file = (Path(args.data) / args.output).with_suffix('.pth')
        torch.save(model.state_dict(), save_file)
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Saved weights to {save_file}')
        self._plot_losses(train_loss, test_loss, args.figs, args.output, args.NIterations,best_epoch,
                          title=f'{args.input[0]}: half window size={examples.window_size}, embedding dimension={args.embedding_dim}')

    def _plot_losses(self, training_losses: [float], test_losses: [float], figs: str, output: str, NIterations: int,best_epoch:int,
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
        ax1.plot(list(range(1, len(training_losses) + 1)), test_losses, label=f'Test {test_losses[-1]:.6}', c='xkcd:red')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        _, ymax = ax1.get_ylim()
        ax1.set_ylim(0, ymax)
        ax1.vlines(best_epoch,0,ymax,colors='xkcd:red',linestyles='dashed',label='Best')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(title='Losses')
        if title != None:
            ax1.set_title(title)
            
        fig.savefig((Path(figs) / output).with_suffix('.png'))

def probability(s):
    '''
    Function used to validate input
    
    Returns:
       Value, converted to float, provided it is within range
    '''
    p = float(s)
    if 0.0 < p and p < 1.0:
        return p
    else:
        raise ValueError(f'Probability is {p}, should be in range (0,1)')
        
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
    
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=choices, help='Selects the function that is to be executed')
    parser.add_argument('input', nargs='+', help='List of input files')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generation')
    parser.add_argument('--data', default=data, help=f'Path to data files [{data}]')
    parser.add_argument('--logs', default=logs, help=f'Location for storing log files [{logs}]')
    parser.add_argument('-o', '--output', default=None, required=True, help='File name for storing results')
    
    group_create_examples = parser.add_argument_group('examples','Options for creating training examples')
    group_create_examples.add_argument('-n', '--window_size', type=int, default=window_size, 
                                    help=f'Half size of window (context extends left and right) [{window_size}]')
    group_create_examples.add_argument('--test_set_size',type=probability,default=test_set_size,
                                       help='Fraction of dataset that becomes test set')
    
    group_train_embeddings = parser.add_argument_group('train','Options for Training')
    group_train_embeddings.add_argument('-D', '--embedding_dim', type=int, default=300,help='Dimensionality of Embedding vectors')
    group_train_embeddings.add_argument('-N', '--NIterations', type=int, default=100,
                                        help='Number of epochs for training')
    group_train_embeddings.add_argument('--burn',type=int,default=5,help='Burn in')
    group_train_embeddings.add_argument('--batch', type=int, default=batch,help='Batch size for training')
    group_train_embeddings.add_argument('--reload', default=None,
                                        help='Indicates that weights are to be reloaded from file before training')
    group_train_embeddings.add_argument('--show', default=False, action='store_true', help='Controls whether plots are shown')
    group_train_embeddings.add_argument('--figs', default=figs, help=f'Path used to store plots [{figs}]')    
    
    return parser.parse_args()


def train(
        model: 'WordEmbeddings',
        train_dataloader: 'DataLoader' = None,
        test_dataloader: 'DataLoader' = None,
        NIterations: int = 100,
        burn: int=5,
        file_for_best:str = 'best'
    ) -> [float]:
    '''
    Train model and compute tarining and test losses
    
    Parameters:
        model
        train_dataloader
        test_dataloader
        NIterations
        
    Returns:
       Training losses for each epoch
       Test Losses for each epoch
    '''

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.95)
    loss_fn = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    running_train_loss = []
    running_test_loss = []
    best_epoch = -1
    for epoch in range(NIterations):       
        train_loss = []
        model.train()

        for feature, label in train_dataloader:
            y_train_pred = model(feature.to(device))
            loss = loss_fn(y_train_pred, label.to(device))
            train_loss.append(loss.item() * feature.size(0)) # Scale by batch size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_train_loss = np.sum(train_loss) / len(train_dataloader)
        running_train_loss.append(mean_train_loss)

        test_loss = []
        model.eval()
        with torch.no_grad():
            for feature, label in test_dataloader:
                y_train_pred = model(feature.to(device))
                loss = loss_fn(y_train_pred, label.to(device))
                test_loss.append(loss.item() * feature.size(0))

        mean_test_loss = np.sum(test_loss) / len(test_dataloader)
        running_test_loss.append(mean_test_loss)

        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Epoch: {epoch}, Mean Training Loss = {mean_train_loss}, Mean Test Loss = {mean_test_loss}')
   
       
        if len(running_test_loss) > burn and running_test_loss[-1] < np.average(running_test_loss[-burn:-1]):
            torch.save(model.state_dict(), file_for_best)
            best_epoch = epoch
            
        if user_has_requested_stop():
            Logger.get_instance().log(f'{__file__} {Logger.get_line()} Stopping')
            break 
 
    return running_train_loss, running_test_loss,best_epoch


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
