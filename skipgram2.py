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

'''Skipgrams as described in Chapter 6 of Jurafsky & Martin'''

from abc import ABC,abstractmethod
from argparse import ArgumentParser
from glob import glob
from os.path import join
from pathlib import Path
from pickle import dump, HIGHEST_PROTOCOL, load
from shutil import copyfile
from time import time
import numpy as np
from matplotlib.pyplot import figure,show
from matplotlib import rc
from scipy.special import expit
from vocabulary import Vocabulary
from tokenizer import generate_sentences,generate_text,generate_tokens,Token
from shared.utils import Logger, user_has_requested_stop

class Examples:
    '''
    A collection of positive and negative training examplkes.
    
    Attributes:
        k
        vocabulary
        positives
        negatives
        rng
        window

    '''
    
    @staticmethod
    def create(file_name,logger):
        '''
        A factory method to instantiate a set of Examples from a saved file
        
        Parameters:
            file_name    Name of file where examples have been stored
        '''
        with open(file_name, 'rb') as inp:
            product = load(inp) 
            logger.log(f'Loaded examples from {file_name.resolve()}')
            return product
        
    def __init__(self,window=2,k=2,rng=np.random.default_rng(),alpha=0.75):
        self.window = window
        self.k = k
        self.vocabulary = Vocabulary()
        self.positives = np.zeros((0,2))
        self.negatives = np.zeros((0,2))
        self.rng = rng
        self.alpha = alpha
        
    def build(self,sentence_generator):
        '''
        Build examples using a generator for sentences
        
        Parameters:
            sentence_generator    Used to iterate through sentences in corpus
        '''
        self.positives = self._create_positives(sentence_generator)
        self.negatives = self._create_negatives()
        
    def save(self,file,report=print):
        '''
        Save Examples using pickle.
        
        Parameters:
            file     Name of file where tables will be saved
        '''
        with open(file,'wb') as out:
            dump(self, out, HIGHEST_PROTOCOL)
            report (f'Saved examples in {file.resolve()}')    
        
    def _create_positives(self,sentence_generator):
        positives = []
        for sentence in sentence_generator:
            self._add_sentence(sentence,positives)
        return np.array(positives)
        
    def _add_sentence(self,sentence,positives):
        tokens = [self.vocabulary.tokenize(word) for word in sentence if self._is_word(word)]

        for w,c in self._generate_positive(tokens):
            positives.append([w,c])
        
    '''
    Create positive examples
    '''
    def _generate_positive(self,tokens):
        for i in range(len(tokens)):
            for j in range(-self.window,self.window + 1):
                if j == 0: continue
                if i + j < 0: continue
                if i + j >= len(tokens): continue
                yield tokens[i], tokens[i+j]
                
    '''
    Create negative examples
    '''
    def _create_negatives(self):
        indices = np.argsort(self.positives[:,0])
        self.positives = self.positives[indices,:]
        m,_ = self.positives.shape
        ws,breaks = np.unique(self.positives[:,0],return_index=True)
        breaks = np.hstack([breaks,[m]])
        Product = np.full((m*self.k,2),-1)
        counts = self.vocabulary.get_counts()**self.alpha
        for i in range(len(ws)):
            pos_min = self.window*breaks[i]
            pos_max = self.window*breaks[i+1]
            Product[pos_min:pos_max,0] = ws[i]
            Product[pos_min:pos_max,1] = self.rng.choice(len(self.vocabulary),
                                                         p=self._get_p(
                                                             np.unique(self.positives[breaks[i]:breaks[i+1],1]),counts),
                                                         size=pos_max - pos_min)
        return Product
 
    def _get_p(self,forbidden,counts):
        '''
        Calculate probabilities using equation (6.32)
        
        Parameters:
            forbidden
            counts
                    
        '''
        p = counts.copy()
        p[forbidden] = 0
        return p / p.sum()    
    
    
    def _is_word(self,s):
        '''
        Verify that a string is composed solely of letters and apostophes
        
        Parameters:
            s       The string to be tested
        '''
        return s.replace(Token.Apostrophe,'').replace(Token.Apostrophe2,'').isalpha()    

class SkipGram:
    '''
    Skipgrams after Chaper 6
    
    Attributes:
        examples
        word_vectors
        context_vectors
        rng
        minibatch
    '''
    @staticmethod
    def create_unit_vectors(m,dimensionality,rng):
        '''
        Create a set of normed unit vectors 
        
        Parameters:
            m               Number of vectors
            dimensionality  Dimensionality
            rng             Random number generator
        '''
        Product = rng.uniform(size=(m,dimensionality))
        return Product/np.linalg.norm(Product,axis=1,keepdims=True)
    
    @staticmethod
    def create(file_name,report=print):
        '''
        A factory method to instantiate a Skipgram from a saved file
        
        Parameters:
            file_name    Name of file where ngrams have been stored
        '''
        with open(file_name, 'rb') as inp:
            product = load(inp) 
            report (f'Loaded skipgrams from {file_name.resolve()}')
            return product    
    
    def __init__(self,examples,dimensionality=128,logger=None,rng=np.random.default_rng(),minibatch=1):
        self.examples = examples
        n_words = len(examples.vocabulary)
        self.word_vectors = SkipGram.create_unit_vectors(n_words,dimensionality,rng)
        self.context_vectors = SkipGram.create_unit_vectors(n_words,dimensionality,rng)
        self.rng = rng
        self.minibatch = minibatch
        
    def step(self,loss_calculator,eta=0.01):
        '''
        Compute partial derivatives and update w and c
        Section 6.8.2
        '''
        n,_ = self.examples.positives.shape
        for i in self.rng.integers(n,size=(self.minibatch)):
            w_index = self.examples.positives[i,0]
            c_index = self.examples.positives[i,1]
            w = self.word_vectors[w_index,:]
            c_pos = self.context_vectors[c_index,:]
            
            #  Equation (6.35)
            sigma_w_c_pos = expit(np.dot(w,c_pos))
            dL_dc_pos = (sigma_w_c_pos - 1) * w  
            
            # Equation (6.36)
            dL_dc_neg = []
            sigma_w_c_neg = []
            for j in range(self.examples.k):
                c_neg_index = self.examples.negatives[i*self.examples.k+j,1]
                c_neg = self.context_vectors[c_neg_index,:]
                sigma_w_c_neg.append(expit(np.dot(w,c_neg)))
                dL_dc_neg.append(sigma_w_c_neg[-1] * w) 
            
            # Equation (6.37)    
            dL_dw = (sigma_w_c_pos - 1) * c_pos
            for j in range(self.examples.k):
                c_neg_index = self.examples.negatives[i*self.examples.k+j,1]
                c_neg=  self.context_vectors[c_neg_index,:] 
                dL_dw += sigma_w_c_neg[j] * c_neg
                
            self.context_vectors[c_index,:] -= eta * dL_dc_pos    # (6.38)
            self.word_vectors[w_index,:] -= eta * dL_dw        # (6.39)
            for j in range(self.examples.k):
                c_neg_index = self.examples.negatives[i*self.examples.k+j,1]
                self.context_vectors[c_neg_index,:]  -= eta*dL_dc_neg[j]    # (6.40) 
                
            loss_calculator.append(i)
    
    def save(self,file_path,report=print):
        '''
        Save Examples using pickle.
        
        Parameters:
            file_path     Name of file where tables will be saved
        '''
        if file_path.is_file():
            copyfile(file_path,file_path.with_suffix('.pkl~'))
        with open(file_path,'wb') as out:
            dump(self, out, HIGHEST_PROTOCOL)
            report (f'Saved examples in {file_path.resolve()}')        

    def calculate_products(self):
        '''
        Calculate dot products between word vectors
        '''
        m,n = self.word_vectors.shape
        self.P = np.zeros((m,m))
        for i in range(m):
            for j in range(i+1):
                self.P[i,j] = np.dot(self.word_vectors[i,:],self.word_vectors[j,:])
                self.P[j,i] = self.P[i,j]
    
class LossCalculator:
    '''
    Compute loss following equation (6.34). We will cache the
    losses for each word, so we need merely recalculate the
    losses that have changed.
    
    Attributes:
        positives
        negatives
        k
        word_vectors
        context_vectors
        Losses
        ToCalculate
    '''
    def __init__(self,examples,skipgram):
        self.positives = examples.positives
        self.negatives = examples.negatives
        self.k = examples.k
        self.word_vectors = skipgram.word_vectors
        self.context_vectors = skipgram.context_vectors
        n,_ = self.positives.shape
        self.Losses = np.zeros((n))
        self.ToCalculate = list(range(n))
        
    def append(self,i):
        '''
        Indicate that one word has changed
        '''
        self.ToCalculate.append(i)
        
    def get_loss(self):
        '''
        Compute loss following equation (6.34). After the first
        call we need only recalculate for words that have changed.
        '''
        for i in self.ToCalculate:
            w_index = self.positives[i,0]
            c_index = self.positives[i,1]
            w = self.word_vectors[w_index,:]
            c_pos = self.context_vectors[c_index,:]
            self.Losses[i] = - np.log(expit(np.dot(w,c_pos)))
            for j in range(self.k):
                assert w_index == self.negatives[i*self.k+j,0]
                c_neg_index = self.negatives[i*self.k+j,1]
                c_neg= self.context_vectors[c_neg_index,:]
                self.Losses[i]  -= np.log(expit(-np.dot(w,c_neg)))
                     
        return self.Losses.sum()
    
    def reset(self):
        '''
        Once we have calculated loss, need to reset the list of changes
        '''
        self.ToCalculate = []   
    


class Command(ABC):
    '''
    This class is te parant for all the taks performed by this program.
    It provides a list of ecceptable commands (used by parse_args(),
    and its subclasses are each esponsible for one task.
    '''
    choices = {}
    
    @staticmethod
    def append(commands):
        for command in commands:
            Command.choices[command.key] = command
        
    @staticmethod
    def get_choices():
        return [key for key in Command.choices.keys()]
    
    @staticmethod
    def get_command(key):
        return Command.choices[key]
        
    def __init__(self,key):
        self.key = key
    
    def execute(self,args):
        '''
        Set up parameters needed by Command, then execute it
        '''
        with Logger(Path(__file__).stem,path=args.logs) as logger:
            self._execute(args,rng = np.random.default_rng(args.seed),logger=logger)
        
    @abstractmethod
    def _execute(self,args,rng = np.random.default_rng(),logger=None):
        '''
        Perform command
        '''
    
class CreateExamples(Command):
    '''
    Build examples for training skipgrams after 6.8.2 of Jurafsky & Martin
    '''
    def __init__(self):
        super().__init__('examples')
        
    def _execute(self,args,rng = np.random.default_rng(),logger=None):
        '''
        Parse text into token, then build examples
        ''' 
        examples = Examples(window=args.window,k=args.k,rng=rng)
        examples.build(
            generate_sentences(
                generate_tokens(
                    generate_text(
                        file_names=[globbed for name in args.input for globbed in glob(join(args.data, name))],
                        logger=logger
                    ))))
        
        examples.save((Path(args.data) / args.output).with_suffix('.pkl'),report=lambda s:logger.log(s))        

class TrainSkipgrams(Command):
    '''
    Adjust weights of  skipgrams after 6.8.2 of Jurafsky & Martin
    '''
    def __init__(self):
        super().__init__('train')
        
    def _execute(self,args,rng = np.random.default_rng(),logger=None):
        '''
        Adjust weights of  skipgrams after 6.8.2 of Jurafsky & Martin
        '''
        trainer = SkipGram(Examples.create((Path(args.data) / args.input[0]).with_suffix('.pkl'),logger),
                           dimensionality=args.dimensionality,
                           logger=logger,
                           rng=rng,
                           minibatch=args.minibatch)
        losses = []
        loss_calculator = LossCalculator(trainer.examples,trainer) 
        for i in range(args.Niterations):
            trainer.step(loss_calculator,eta=args.eta)
            losses.append(loss_calculator.get_loss())
            loss_calculator.reset()
            if i % args.freq == 1:
                logger.log (f'Step {i}, loss={losses[-1]}')
                trainer.save((Path(args.data) / args.output).with_suffix('.pkl'),report=lambda s:logger.log(s))
                if user_has_requested_stop(): break
                
        fig = figure(figsize=(10,10))
        ax = fig.add_subplot(1,1,1)
        ax.plot(losses)
        ax.set_title(f'Training: dimensionality={args.dimensionality}, minibatch = {args.minibatch}, '
                     r'$\eta=$'
                     f'{args.eta}')
        y0,y1=ax.get_ylim()
        ax.set_ylim(0,y1)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        
        fig.savefig((Path(args.figs) / args.output).with_suffix('.png'))        
        
class BuildDistances(Command):
    '''
    Build table of scalar products between weight vectors
    '''
    def __init__(self):
        super().__init__('build')
        
    '''
    Build table of scalar products between weight vectors
    '''        
        
    def _execute(self,args,rng = np.random.default_rng(),logger=None):
        skipgram = SkipGram.create((Path(args.data) / args.input[0]).with_suffix('.pkl'),
                                   report=lambda s: logger.log(s))
        logger.log('Calculating products')
        skipgram.calculate_products()
        logger.log('Calculated products')
        skipgram.save((Path(args.data) / args.output).with_suffix('.pkl'),report=lambda s:logger.log(s))
 
class Explore(Command):
    '''
    Build table of scalar products between weight vectors
    '''
    def __init__(self):
        super().__init__('explore')
        
    '''
    Build table of scalar products between weight vectors
    '''        
        
    def _execute(self,args,rng = np.random.default_rng(),logger=None):
        skipgram = SkipGram.create((Path(args.data) / args.input[0]).with_suffix('.pkl'),
                                   report=lambda s: logger.log(s))
        P = skipgram.P
        
        vocabulary = skipgram.examples.vocabulary
        if args.word == None:
            for token in rng.integers((len(vocabulary)),size=args.nwords):
                word = vocabulary.get_word(token)
                indices = np.argsort(P[token,:])[::-1]
                for i in range(args.nclosest):
                    logger.log(f'{token} {word}: {indices[i]} {vocabulary.get_word(indices[i])} {P[token,indices[i]]}')
        else:
            token = vocabulary.token[args.word]
            indices = np.argsort(P[token,:])[::-1]
            for i in range(args.nclosest):
                logger.log(f'{token} {args.word}: {indices[i]} {vocabulary.get_word(indices[i])} {P[token,indices[i]]}')            

def parse_args(choices):
    window = 2
    k = 2
    alpha = 0.75
    dimensionality = 128
    eta = 0.01
    minibatch = 2**12
    Niterations = 10000
    freq = 50
    nwords = 12
    nclosest = 32
    
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=choices)
    parser.add_argument('input',nargs='+',help='List of input files')
    parser.add_argument('--seed',type=int,default=None,help='Deed for random number generation')
    parser.add_argument('--data', default='./data',help='Path to data files') 
    parser.add_argument('-o', '--output',default=None,required=True,help='File name for storing results')
    parser.add_argument('--logs', default='./logs', help='Location for storing log files')
    parser.add_argument('--show', default=False,action='store_true',help='Controls whether plots are shown')
    parser.add_argument('--figs', default='./figs',help='Path used to store plots')        
    
    examples_group = parser.add_argument_group('Examples',description='Used for command=examples')
    examples_group.add_argument('-w','--window', type=int,default=window,help=f'Width of window for context [{window}]')
    examples_group.add_argument('-k','--k',type=int,default=k,help=f'Number of negative context words for each positive [{k}]')
    examples_group.add_argument('--alpha',default=alpha,type=float,help=f'The exponent from equation (6.32) [{alpha}]')
    
    training_group = parser.add_argument_group('Training',description='Used for command==train')
    training_group.add_argument('-d','--dimensionality',type=int,default=dimensionality,help=f'Length of word vectors [{dimensionality}]')
    training_group.add_argument('--eta',default=eta,type=float,help=f'Training speed [{eta}]')
    training_group.add_argument('-m','--minibatch',type=int,default=minibatch,help=f'Number of samples in a minibatch [{minibatch}]')
    training_group.add_argument('-N','--Niterations',type=int,default=Niterations,help=f'Number of iterations [{Niterations}]')
    training_group.add_argument('--freq',type=int,default=freq,help=f'Interval between printing training steps [{freq}]')
    
    analysis_group = parser.add_argument_group(title='Analysis',description='Used for build and explore')
    analysis_group.add_argument('--word',default=None,help='Used to explore a single word')
    analysis_group.add_argument('--nwords',type=int,default=nwords,help=f'Number of words to explore [{nwords}]')
    analysis_group.add_argument('--nclosest',type=int,default=nclosest,help=f'Number of closest words to explore [{nclosest}]')
    
    return parser.parse_args()
        
def main():
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)    
    start  = time()
    Command.append([
        CreateExamples(),
        TrainSkipgrams(),
        BuildDistances(),
        Explore()
    ])
    args = parse_args(Command.get_choices())
    Command.get_command(args.command).execute(args)

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    
    if args.show:
        show()
    
if __name__=='__main__':
    main()
