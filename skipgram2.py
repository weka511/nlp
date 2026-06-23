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
    Skipgrams as described in Chapter 6 of Jurafsky & Martin.
    This program generates examples, train skipgrams, and
    allows the user to explore word vectors.
'''

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
    A collection of positive and negative training examples.
    
    Attributes:
        k             Number of negative context words for each positive
        vocabulary    Relates words to tokens
        positives     Positive examples: pairs w and c, which form inices in vocabulary
        negatives     Negative examples: pairs w and c, which form inices in vocabulary
        rng           Random number generator
        window        Width of window for context
    '''
    
    @staticmethod
    def create(file_name,logger):
        '''
        A factory method to instantiate a set of Examples from a saved file
        
        Parameters:
            file_name    Name of file where examples have been stored
            logger
        '''
        with open(file_name, 'rb') as inp:
            product = load(inp) 
            logger.log(f'Loaded examples from {file_name.resolve()}')
            return product
        
    def __init__(self,window=2,k=2,rng=np.random.default_rng(),alpha=0.75):
        '''
        Parameters:
            window    Width of window for context
            k         Number of negative context words for each positive
            rng       Random number generator
            alpha     The exponent from equation (6.32) 
        '''
        self.window = window
        self.k = k
        self.vocabulary = Vocabulary()
        self.positives = np.zeros((0,2))
        self.negatives = np.zeros((0,2))
        self.rng = rng
        self.alpha = alpha
        
    def build(self,sentence_generator,logger=None):
        '''
        Build examples using a generator for sentences
        
        Parameters:
            sentence_generator    Used to iterate through sentences in corpus
        '''
        logger.log(f'Building examples: window={self.window}, k={self.k}')
        self.positives = self._create_positives(sentence_generator)
        logger.log('Built positive examples')
        self.negatives = self._create_negatives()
        logger.log('Completed building examples')
        
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
        '''
        Create array of positive examples
        
        Parameters:
            sentence_generator   Used to extract tokens
            
        Returns:
            Matrix of negative examples
        '''
        positives = []
        for sentence in sentence_generator:
            self._add_sentence(sentence,positives)
        return np.array(positives)
        
    def _add_sentence(self,sentence,positives):
        '''
        Add one sentance to positives
        
        Parameters:
            sentence
            positives
        '''
        tokens = [self.vocabulary.tokenize(word) for word in sentence if self._is_word(word)]

        for w,c in self._generate_positive(tokens):
            positives.append([w,c])
        
    '''
    Create positive examples. Each token gines rise to a set of pairs: the second
    token of each pair is close to the first, within a distance controlled by self.window.
    
    Parameters:
        tokens    List of tokens for positive examples
    '''
    def _generate_positive(self,tokens):
        for i in range(len(tokens)):
            for j in range(-self.window,self.window + 1):
                if j != 0:    # Pair of tokens must not be identical
                    try:
                        yield tokens[i], tokens[i+j]
                    except IndexError:               # Don't go beyond boundary of tokens
                        pass
                
    '''
    Create negative examples. For each positive example, (w,c), create k pairs (w,c'),
    where c' is never used in any positve example for w.
    
    Returns:
         Matrix of negative examples
    '''
    def _create_negatives(self):
        indices = np.argsort(self.positives[:,0])
        self.positives = self.positives[indices,:]
        m,_ = self.positives.shape
        word_tokens,breaks = np.unique(self.positives[:,0],return_index=True)
        breaks = np.hstack([breaks,[m+1]])
        Product = np.full((m*self.k,2),-1)
        for i in range(len(word_tokens)):
            neg_min = self.k*breaks[i]                  # Position of first negative example for ith token
            neg_max = min(self.k*breaks[i+1],m*self.k)  # Just beyond last negative example for ith token
            n_negative = neg_max - neg_min              # Number of examples to be generated this step
            Product[neg_min:neg_max,0] = word_tokens[i] # Word for negatve examples must match postive
            
            probabilities = self._get_p(forbidden=np.unique(self.positives[breaks[i]:breaks[i+1],1]),
                                        counts=self.vocabulary.get_counts())  
            negative_contexts = self.rng.choice(len(self.vocabulary),p=probabilities,size=n_negative)
            Product[neg_min:neg_max,1] = negative_contexts
        return Product
 
    def _get_p(self,forbidden=[],counts=[]):
        '''
        Use equation (6.32) to calculate probabilities of each token appearing in negative context
        
        Parameters:
            forbidden    Tokens that appear as context in positive examples, 
                         so we cannot use them for negatives
            counts       Number of appearance of each token in vocabulary     
                    
        '''
        p = counts**self.alpha
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
        batch
    '''
    @staticmethod
    def create_unit_vectors(m,ndim,rng):
        '''
        Create a set of normed unit vectors 
        
        Parameters:
            m               Number of vectors
            ndim  Dimensionality
            rng             Random number generator
        '''
        Product = rng.uniform(size=(m,ndim))
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
    
    def __init__(self,examples,ndim=128,logger=None,rng=np.random.default_rng(),batch=1):
        self.examples = examples
        n_words = len(examples.vocabulary)
        self.word_vectors = SkipGram.create_unit_vectors(n_words,ndim,rng)
        self.context_vectors = SkipGram.create_unit_vectors(n_words,ndim,rng)
        self.rng = rng
        self.batch = batch
        
    def step(self,loss_calculator,eta=0.01):
        '''
        Compute partial derivatives and update w and c
        Section 6.8.2
        '''
        n,_ = self.examples.positives.shape
        for i in self.rng.integers(n,size=(self.batch)):
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
    
    def save(self,file_path,report=print,description='word vectors'):
        '''
        Save Examples using pickle.
        
        Parameters:
            file_path     Name of file where tables will be saved
            report        Used to log result of save
            description   Name reported to user
        '''
        if file_path.is_file():
            copyfile(file_path,file_path.with_suffix('.pkl~'))
        with open(file_path,'wb') as out:
            dump(self, out, HIGHEST_PROTOCOL)
            report (f'Saved {description} in {file_path.resolve()}')        

    def calculate_products(self,normalize):
        '''
        Calculate dot products between word vectors
        
        Parameters:
            normalize   Normalize verctors when calculating product
        '''
        m,n = self.word_vectors.shape
        norms = np.linalg.norm(self.word_vectors,axis=1) if normalize else np.ones((m))    
        self.P = np.zeros((m,m))
        for i in range(m):
            for j in range(i+1):
                self.P[i,j] = np.dot(self.word_vectors[i,:],self.word_vectors[j,:]) / (norms[i]*norms[j])                
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
                    ))),
            logger=logger)
        
        examples.save((Path(args.data) / args.output).with_suffix('.pkl'),report=lambda s:logger.log(s))        

class AbstractTrainer(Command):
    '''
    Adjust weights of  skipgrams after 6.8.2 of Jurafsky & Martin
    '''
    def __init__(self,key):
        super().__init__(key)
        
    @abstractmethod    
    def _create(self,args,rng = np.random.default_rng(),logger=None):
        ...

        #return SkipGram(Examples.create((Path(args.data) / args.input[0]).with_suffix('.pkl'),logger),
                        #ndim=args.ndim,
                        #logger=logger,
                        #rng=rng,
                        #batch=args.batch)
        
    def _execute(self,args,rng = np.random.default_rng(),logger=None):
        '''
        Adjust weights of  skipgrams after 6.8.2 of Jurafsky & Martin
        '''
        trainer = self._create(args,rng,logger)
        losses = []
        loss_calculator = LossCalculator(trainer.examples,trainer) 
        for i in range(args.Niter):
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
        ax.set_title(f'Training: ndim={args.ndim}, batch = {args.batch}, '
                     r'$\eta=$'
                     f'{args.eta}')
        y0,y1=ax.get_ylim()
        ax.set_ylim(0,y1)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        
        fig.savefig((Path(args.figs) / args.output).with_suffix('.png'))        

class TrainSkipgrams(AbstractTrainer):
    '''
    Adjust weights of  skipgrams after 6.8.2 of Jurafsky & Martin
    '''
    def __init__(self):
        super().__init__('train')
        
    def _create(self,args,rng = np.random.default_rng(),logger=None):

        return SkipGram(Examples.create((Path(args.data) / args.input[0]).with_suffix('.pkl'),logger),
                        ndim=args.ndim,
                        logger=logger,
                        rng=rng,
                        batch=args.batch)
    
class RestartSkipgrams(AbstractTrainer):
    '''
    Adjust weights of  skipgrams after 6.8.2 of Jurafsky & Martin
    '''
    def __init__(self):
        super().__init__('restart')
        
    def _create(self,args,rng = np.random.default_rng(),logger=None):
        logger.log(f'Restarting Training')
        return SkipGram.create((Path(args.data) / args.input[0]).with_suffix('.pkl'),
                                   report=lambda s: logger.log(s))
 
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
        skipgram.calculate_products(args.normalize)
        logger.log('Calculated products')
        skipgram.save((Path(args.data) / args.output).with_suffix('.pkl'),
                      report=lambda s:logger.log(s),description='word vectors and distances')
 
class Explore1(Command):
    '''
    Select entries from table of scalar products
    '''
    def __init__(self):
        super().__init__('explore1')
        
    '''
    Select entries from table of scalar products
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


class Explore2(Command):
    '''
    Select entries from table of scalar products
    '''
    def __init__(self):
        super().__init__('explore2')
        
    '''
    Select entries from table of scalar products
    '''        
        
    def _execute(self,args,rng = np.random.default_rng(),logger=None):
        skipgram = SkipGram.create((Path(args.data) / args.input[0]).with_suffix('.pkl'),
                                   report=lambda s: logger.log(s))
        P = skipgram.P
        vocabulary = skipgram.examples.vocabulary
        m,n = P.shape
        for i in range(m):
            for j in range(n):
                if i != j and P[i,j] > args.min:
                    logger.log(f'{vocabulary.get_word(i)} {vocabulary.get_word(j)} {P[i,j]}')
 

def parse_args(choices):
    
    # Establish defaults
    
    window = 2
    k = 2
    alpha = 0.75
    ndim = 128
    eta = 0.01
    batch = 2**12
    Niter = 10000
    freq = 50
    nwords = 12
    nclosest = 32
    data = './data'
    logs = './logs'
    figs = './figs'
    threshold = 0.65
    
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=choices,help='Selects the function that is to be executed')
    parser.add_argument('input',nargs='+',help='List of input files')
    parser.add_argument('--seed',type=int,default=None,help='Seed for random number generation')
    parser.add_argument('--data', default=data,help=f'Path to data files [{data}]') 
    parser.add_argument('-o', '--output',default=None,required=True,help='File name for storing results')
    parser.add_argument('--logs', default=logs, help=f'Location for storing log files [{logs}]')
    parser.add_argument('--show', default=False,action='store_true',help='Controls whether plots are shown')
    parser.add_argument('--figs', default=figs,help=f'Path used to store plots [{figs}]')        
    
    examples_group = parser.add_argument_group('Examples',description='Used for examples')
    examples_group.add_argument('-w','--window', type=int,default=window,help=f'Width of window for context [{window}]')
    examples_group.add_argument('-k','--k',type=int,default=k,help=f'Number of negative context words for each positive [{k}]')
    examples_group.add_argument('--alpha',default=alpha,type=float,help=f'The exponent from equation (6.32) [{alpha}]')
    
    training_group = parser.add_argument_group('Training',description='Used for train')
    training_group.add_argument('-d','--ndim',type=int,default=ndim,help=f'Length of word vectors [{ndim}]')
    training_group.add_argument('--eta',default=eta,type=float,help=f'Training speed [{eta}]')
    training_group.add_argument('-m','--batch',type=int,default=batch,help=f'Number of samples in a batch [{batch}]')
    training_group.add_argument('-N','--Niter',type=int,default=Niter,help=f'Number of iterations [{Niter}]')
    training_group.add_argument('--freq',type=int,default=freq,help=f'Interval between printing training steps [{freq}]')
    
    build_group = parser.add_argument_group(title='Build',description='Used for building distances')
    build_group.add_argument('--normalize', default=False,action='store_true',
                             help='Normalize vectors before calculating products')
    
    explore_group = parser.add_argument_group(title='Explore1',description='Used for explore1')
    explore_group.add_argument('--word',default=None,help='Used to explore a single word')
    explore_group.add_argument('--nwords',type=int,default=nwords,help=f'Number of words to explore [{nwords}]')
    explore_group.add_argument('--nclosest',
                               type=int,default=nclosest,help=f'Number of closest words to explore [{nclosest}]')
    explore_group2 = parser.add_argument_group(title='Explore2',description='Used for explore2')
    explore_group2.add_argument('--min',type=float,default=threshold,
                               help=f'Display pairs if product exceeds this value [{threshold}]')
    
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
        RestartSkipgrams(),
        BuildDistances(),
        Explore1(),
        Explore2(),
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
