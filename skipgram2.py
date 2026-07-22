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
    Skipgrams as described in Chapter 6 of 	
    Speech and Language Processing (3rd ed. draft)
    Dan Jurafsky and James H. Martin .
    This program generates examples, trains skipgrams, and
    allows the user to explore word vectors.
'''

__version__ = '0.0'
__author__ = 'Simon Crase'

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from glob import glob
from os.path import join
from pathlib import Path
from pickle import dump, HIGHEST_PROTOCOL, load
from shutil import copyfile
from time import time

import numpy as np
from matplotlib.pyplot import figure, show
from matplotlib import rc
from scipy.special import expit
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

from crp import ChineseRestaurantProcess, DistanceDependentChooser
from vocabulary import Vocabulary
from tokenizer import generate_sentences, generate_text, generate_tokens, Token
from shared.utils import Logger, user_has_requested_stop, get_seed


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
    def create(file_name):
        '''
        A factory method to instantiate a set of Examples from a saved file
        
        Parameters:
            file_name    Name of file where examples have been stored
        '''
        with open(file_name, 'rb') as inp:
            product = load(inp)
            Logger.get_instance().log(f'{__file__} {Logger.get_line()} Loaded examples from {file_name.resolve()}')
            return product

    def __init__(self, window=2, k=2, rng=np.random.default_rng(), alpha=0.75):
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
        self.positives = np.zeros((0, 2))
        self.negatives = np.zeros((0, 2))
        self.rng = rng
        self.alpha = alpha

    def build(self, sentence_generator):
        '''
        Build examples using a generator for sentences
        
        Parameters:
            sentence_generator    Used to iterate through sentences in corpus
        '''
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Building examples: window={self.window}, k={self.k}')
        self.positives = self._create_positives(sentence_generator)
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Built positive examples')
        self.negatives = self._create_negatives()
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Completed building examples')

    def save(self, file, report=lambda s: Logger.get_instance().log(f'{__file__} {Logger.get_line()} {s}')):
        '''
        Save Examples using pickle.
        
        Parameters:
            file     Name of file where tables will be saved
        '''
        with open(file, 'wb') as out:
            dump(self, out, HIGHEST_PROTOCOL)
            report(f'Saved examples in {file.resolve()}')

    def _create_positives(self, sentence_generator):
        '''
        Create array of positive examples
        
        Parameters:
            sentence_generator   Used to extract tokens
            
        Returns:
            Matrix of negative examples
        '''
        positives = []
        for sentence in sentence_generator:
            self._add_sentence(sentence, positives)
        return np.array(positives)

    def _add_sentence(self, sentence, positives):
        '''
        Add pairs of tokens, (word,context), generated from sentance to positives
        
        Parameters:
            sentence    A sentence in text form (not tokenized)
            positives   A list of positive examples, i.e. Pairs of tokens, (word,context)
        '''
        tokens = [self.vocabulary.tokenize(word) for word in sentence if self._is_word(word)]

        for w, c in self._generate_positive(tokens):
            positives.append([w, c])

    '''
    Create positive examples. Each token gives rise to a set of pairs: the second
    token of each pair is from a position in the sentence close to the first,
    within a distance controlled by self.window.
    
    Parameters:
        tokens    List of tokens for positive examples
    '''

    def _generate_positive(self, tokens):
        for i in range(len(tokens)):
            for j in range(-self.window, self.window + 1):
                if j != 0:    # Pair of tokens must not be identical
                    try:
                        yield tokens[i], tokens[i + j]
                    except IndexError:               # Don't go beyond boundary of tokens
                        pass

    '''
    Create negative examples. For each positive example, (w,c), create k pairs (w,c'),
    where c' is never used in any positve example for w.
    
    Returns:
         Matrix of negative examples
    '''

    def _create_negatives(self):
        indices = np.argsort(self.positives[:, 0])
        self.positives = self.positives[indices, :]
        m, _ = self.positives.shape
        word_tokens, breaks = np.unique(self.positives[:, 0], return_index=True)
        breaks = np.hstack([breaks, [m + 1]])
        Product = np.full((m * self.k, 2), -1)
        for i in range(len(word_tokens)):
            neg_min = self.k * breaks[i]                  # Position of first negative example for ith token
            neg_max = min(self.k * breaks[i + 1], m * self.k)  # Just beyond last negative example for ith token
            n_negative = neg_max - neg_min              # Number of examples to be generated this step
            Product[neg_min:neg_max, 0] = word_tokens[i] # Word for negatve examples must match postive

            probabilities = self._get_p(forbidden=np.unique(self.positives[breaks[i]:breaks[i + 1], 1]),
                                        counts=self.vocabulary.get_counts())
            negative_contexts = self.rng.choice(len(self.vocabulary), p=probabilities, size=n_negative)
            Product[neg_min:neg_max, 1] = negative_contexts
        return Product

    def _get_p(self, forbidden=[], counts=[]):
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

    def _is_word(self, s):
        '''
        Verify that a string is composed solely of letters and apostophes
        
        Parameters:
            s       The string to be tested
        '''
        return s.replace(Token.Apostrophe, '').replace(Token.Apostrophe2, '').isalpha()


class SkipGram:
    '''
    Skipgrams after Chaper 6 of	Jurafsky and Martin 
    
    Attributes:
        examples         Training examples
        word_vectors     Word vectors produced by skipgram training
        context_vectors  Context vectors produced by skipgram training
        rng              Random number generator
        batch            Batch size
    '''
    @staticmethod
    def create_unit_vectors(m, ndim, rng):
        '''
        Create a set of normed unit vectors 
        
        Parameters:
            m       Number of vectors
            ndim    Dimensionality
            rng     Random number generator
        '''
        Product = rng.uniform(size=(m, ndim))
        return Product / np.linalg.norm(Product, 
                                        axis=1,
                                        keepdims=True)

    @staticmethod
    def create(file_name,
               report=lambda s: Logger.get_instance().log(f'{__file__} {Logger.get_line()} {s}')):
        '''
        A factory method to instantiate a Skipgram from a saved file
        
        Parameters:
            file_name    Name of file where ngrams have been stored
            report       Function used to report success
        '''
        with open(file_name, 'rb') as inp:
            product = load(inp)
            report(f'Loaded skipgrams from {file_name.resolve()}')
            return product

    def __init__(self, examples, ndim=128, rng=np.random.default_rng(), batch=1):
        '''
        Parameters:
            examples   Training examples
            ndim       Size of word vectors
            rng        Random number generator
            batch      Batch size
        '''
        self.examples = examples
        n_words = len(examples.vocabulary)
        self.word_vectors = SkipGram.create_unit_vectors(n_words, ndim, rng)
        self.context_vectors = SkipGram.create_unit_vectors(n_words, ndim, rng)
        self.rng = rng
        self.batch = batch

    def step(self, loss_calculator, eta=0.01):
        '''
        Compute partial derivatives and update w and c
        Section 6.8.2
        
        Parameters:
            loss_calculator    Used to calculate loss
            eta                Training rate
        '''
        n, _ = self.examples.positives.shape
        for i in self.rng.integers(n, size=(self.batch)):
            w_index = self.examples.positives[i, 0]
            c_index = self.examples.positives[i, 1]
            w = self.word_vectors[w_index, :]
            c_pos = self.context_vectors[c_index, :]

            #  Equation (6.35)
            sigma_w_c_pos = expit(np.dot(w, c_pos))
            dL_dc_pos = (sigma_w_c_pos - 1) * w

            # Equation (6.36)
            dL_dc_neg = []
            sigma_w_c_neg = []
            for j in range(self.examples.k):
                c_neg_index = self.examples.negatives[i * self.examples.k + j, 1]
                c_neg = self.context_vectors[c_neg_index, :]
                sigma_w_c_neg.append(expit(np.dot(w, c_neg)))
                dL_dc_neg.append(sigma_w_c_neg[-1] * w)

            # Equation (6.37)
            dL_dw = (sigma_w_c_pos - 1) * c_pos
            for j in range(self.examples.k):
                c_neg_index = self.examples.negatives[i * self.examples.k + j, 1]
                c_neg = self.context_vectors[c_neg_index, :]
                dL_dw += sigma_w_c_neg[j] * c_neg

            self.context_vectors[c_index, :] -= eta * dL_dc_pos    # (6.38)
            self.word_vectors[w_index, :] -= eta * dL_dw        # (6.39)
            for j in range(self.examples.k):
                c_neg_index = self.examples.negatives[i * self.examples.k + j, 1]
                self.context_vectors[c_neg_index, :] -= eta * dL_dc_neg[j]    # (6.40)

            loss_calculator.append(i)

    def save(self, file_path,
             report=lambda s: Logger.get_instance().log(f'{__file__} {Logger.get_line()} {s}'),
             description='word vectors'):
        '''
        Save Examples using pickle.
        
        Parameters:
            file_path     Name of file where tables will be saved
            report        Used to log result of save
            description   Name reported to user
        '''
        if file_path.is_file():
            copyfile(file_path, file_path.with_suffix('.pkl~'))
        with open(file_path, 'wb') as out:
            dump(self, out, HIGHEST_PROTOCOL)
            report(f'Saved {description} in {file_path.resolve()}')

    def calculate_products(self, normalize):
        '''
        Calculate dot products between word vectors
        
        Parameters:
            normalize   Normalize verctors when calculating product
        '''
        m, n = self.word_vectors.shape
        norms = np.linalg.norm(self.word_vectors, axis=1) if normalize else np.ones((m))
        self.P = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1):
                self.P[i, j] = np.dot(self.word_vectors[i, :], self.word_vectors[j, :]) / (norms[i] * norms[j])
                self.P[j, i] = self.P[i, j]


class LossCalculator:
    '''
    Compute loss following equation (6.34). We will cache the
    losses for each word, so we need merely recalculate the
    losses that have changed.
    
    Attributes:
        positives         Positive examples
        negatives         Negative examples
        k                 Number of negative context words for each positive
        word_vectors      Word vectors learned during training
        context_vectors   Conrext vectors learned during training
        Losses            Used to hold calculated losses
        ToCalculate       Indices of words have changed
    '''

    def __init__(self, examples, skipgram):
        '''
        Parameters:
            examples     Training examples
            skipgram     Skipgram object
        '''
        self.positives = examples.positives
        self.negatives = examples.negatives
        self.k = examples.k
        self.word_vectors = skipgram.word_vectors
        self.context_vectors = skipgram.context_vectors
        n, _ = self.positives.shape
        self.Losses = np.zeros((n))
        self.ToCalculate = list(range(n))

    def append(self, index):
        '''
        Indicate that one word has changed
        
        Parameters:
            index     Identifies word that has changed
        '''
        self.ToCalculate.append(index)

    def get_loss(self):
        '''
        Compute loss following equation (6.34). After the first
        call we need only recalculate for words that have changed.
        '''
        for i in self.ToCalculate:
            w_index = self.positives[i, 0]
            c_index = self.positives[i, 1]
            w = self.word_vectors[w_index, :]
            c_pos = self.context_vectors[c_index, :]
            self.Losses[i] = - np.log(expit(np.dot(w, c_pos)))
            for j in range(self.k):
                assert w_index == self.negatives[i * self.k + j, 0]
                c_neg_index = self.negatives[i * self.k + j, 1]
                c_neg = self.context_vectors[c_neg_index, :]
                self.Losses[i] -= np.log(expit(-np.dot(w, c_neg)))

        return self.Losses.sum()

    def reset(self):
        '''
        Once we have calculated loss, need to reset the list of changes
        '''
        self.ToCalculate = []


class Command(ABC):
    '''
    This class is the parent for all the tasks performed by this program.
    It provides a list of ecceptable commands (used by parse_args(),
    and its subclasses are each esponsible for one task.
    '''
    choices = {}

    @staticmethod
    def append(commands):
        '''
        List of Mommands that are availavle to user
        '''
        for command in commands:
            Command.choices[command.key] = command

    @staticmethod
    def get_choices():
        '''
        Get list of available Commands
        '''
        return [key for key in Command.choices.keys()]

    @staticmethod
    def get_command(key):
        '''
        Get Command given the user's specification
        
        Parameters:
            key      Command name specified by user
        '''
        return Command.choices[key]

    def __init__(self, key):
        '''
        Used when command is created to set the name that the user specifies
        
        Parameters:
            key      The name that the user specifies for this command
        '''
        self.key = key

    def execute(self, args):
        '''
        Set up parameters needed by Command, then execute it
        
        Parameters:
            args    Command line parameters as parsed by parse_args()
        '''
        with Logger(Path(__file__).stem, path=args.logs) as _:
            start = time()
            for key, value in vars(args).items():
                Logger.get_instance().log(f'{__file__} {Logger.get_line()} {key} = {value}')

            seed = get_seed(args.seed,
                            notify=lambda s: Logger.get_instance().log(f'{__file__} {Logger.get_line()}'
                                                                       f' Created new seed {s}'))
            self._execute(args, rng=np.random.default_rng(seed))

            elapsed = time() - start
            minutes = int(elapsed / 60)
            seconds = elapsed - 60 * minutes
            Logger.get_instance().log(f'{__file__} {Logger.get_line()} Elapsed Time {minutes} m {seconds:.2f} s')

    @abstractmethod
    def _execute(self, args, rng=np.random.default_rng()):
        '''
        Perform command
        
        Parameters:
            args       Command line parameters as parsed by parse_args()
            rng        Random number generator
        '''


class CreateExamples(Command):
    '''
    Build examples for training skipgrams after 6.8.2 of Jurafsky & Martin
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
        examples = Examples(window=args.window, k=args.k, rng=rng)

        examples.build(
            generate_sentences(
                generate_tokens(
                    generate_text(
                        file_names=[globbed for name in args.input for globbed in glob(join(args.data, name))]
                    ))))

        examples.save((Path(args.data) / args.output).with_suffix('.pkl'),
                      report=lambda s: Logger.get_instance().log(f'{__file__} {Logger.get_line()} {s}'))

class Stepsize(ABC):
    '''
    This class is responsible for setting the step size, eta
    '''
    @abstractmethod
    def get_eta(k):
        '''
        Assign step size
        
        Parameters:
             k        Iteration number
        '''
        
class AdaptiveStepsize(Stepsize):
    '''
    This class reduces the step size if the loss increases.
    '''
    def __init__(self,args,losses):
        self.eta = args.eta[0]
        self.losses = losses
        self.ratio = args.ratio
        self.eta_min = args.eta[1]
        
    def __str__(self):
        return f'Adaptive Stepsize eta={self.eta}, eta_min={self.eta_min}, ratio={self.ratio}'
        
    def get_eta(self,k):
        '''
        Assign step size. If error is incrasingm reduce stepsize, as long as it exceeds the minimum allowed.
        
        Parameters:
             k        Iteration number
        '''        
        if len(self.losses) > 2 and self.losses[-1] > self.losses[-2] and self.eta > self.eta_min:
            self.eta *= self.ratio
            Logger.get_instance().log(f'{__file__} {Logger.get_line()} Step {k}, eta={self.eta}')

        return self.eta
  

class ReducingStepsize(Stepsize):
    '''
    This class Reduces the step size steadily from maximum to minimum, 
    regardless of loss
    '''
    def __init__(self,args):
        self.eta = args.eta
        self.tau = args.tau
        
    def __str__(self):
        return f'Reducing Stepsize eta={self.eta}, tau={self.tau}'    
        
    def get_eta(self,k):
        '''
        Establish step size
        
        Parameters:
            k       Iteration number
        '''
        if len(self.eta) == 1:
            return self.eta[0]
        elif k < self.tau:
            alpha = k / self.tau
            return (1 - alpha)*self.eta[0] + alpha*self.eta[1]
        else:
            return self.eta[1]
        
class SGD:
    '''
    Minimize losses of skipgrams by
    performing Scalar Gradient Descent
    
    Parameters:
        loss_calculator   Used to calculate loss at each step
        skipgram          Skipgram weights
        Niter             Number of iterations
        freq              Determine jhow often we report progress
        data              Path to data files
        output            Used to construct file name for saibg data
        losses            Used to hold history of losses
        etas              Used to heold history of step sizes
    '''
    
    @staticmethod
    def adjust_args(args):
        '''
        Used to verify arguments that establish step size;
        this is beyond what ArgumentParser can do.
        '''
        match len(args.eta):
            case 1:
                args.tau = 1
            case 2:
                if args.tau == None:
                    args.tau = args.Niter           
            case _:
                raise ValueError(f'Too many values: {args.eta}')
        return args
    
    def __init__(self,loss_calculator, skipgram, args):
        '''
        Initialize Stochastic Gradient Descent
        
        Paramaters
            loss_calculator     Used to calculate loss at each step
            skipgram            Skipgram weights
            args                Command line arguments
        '''
        self.loss_calculator = loss_calculator
        self.skipgram = skipgram
        self.Niter = args.Niter
        self.freq = args.freq
        self.data = args.data
        self.output = args.output
        self.losses = []
        self.etas = []
        match args.stepsize:
            case 'Adaptive':
                self.stepsize = AdaptiveStepsize(args,self.losses)
            case 'Reducing':
                self.stepsize = ReducingStepsize(args)   
                
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Step strategy {self.stepsize}')
        
    def train(self):
        '''
        Adjust weights of  skipgrams after 6.8.2 of Jurafsky & Martin
        
        Returns:
            History of losses
            History of step sizes
        '''
        for k in range(self.Niter):
            self.etas.append(self.stepsize.get_eta(k)) 
            self.skipgram.step(self.loss_calculator, eta=self.etas[-1])
            self.losses.append(self.loss_calculator.get_loss())
            self.loss_calculator.reset()
            if k % self.freq == 1:
                Logger.get_instance().log(f'{__file__} {Logger.get_line()} Step {k}, eta={self.etas[-1]}, loss={self.losses[-1]}')
                self.save_progress()
                if user_has_requested_stop():
                    break

        return self.losses,self.etas 
    
     
    def save_progress(self):
        '''
        Save current state of data
        '''
        self.skipgram.save((Path(self.data) / self.output).with_suffix('.pkl'),
                      report=lambda s: Logger.get_instance().log(f'{__file__} {Logger.get_line()} {s}'))        
    
class AbstractTrainSkipgrams(Command):
    '''
    Used by TrainSkipgrams and RestartSkipgrams to adjust 
    weights of  skipgrams after 6.8.2 of Jurafsky & Martin.
    '''

    def __init__(self, key):
        super().__init__(key)

    def _execute(self, args, rng=np.random.default_rng()):
        '''
        Adjust weights of  skipgrams and plot losses
        
        Parameters:
            args       Command line parameters as parsed by parse_args()
            rng        Random number generator
        '''
        losses,etas = self._train(self._create(args, rng),args)
        self._plot_losses(args, losses,etas)

    @abstractmethod
    def _create(self, args, rng=np.random.default_rng()):
        '''
        Implemented by descendents to either instantiate a new Skipgram
        or recall one that has been created previously.
        
        Parameters:
            args       Command line parameters as parsed by parse_args()
            rng        Random number generator
        '''

    def _train(self, skipgram, args):
        '''
        Adjust weights of  skipgrams after 6.8.2 of Jurafsky & Martin
        
        Parameters:
            skipgram    The Skipgram to be trained
            args        Command line parameters as parsed by parse_args()
        '''
  
        return SGD(
            LossCalculator(skipgram.examples,
                           skipgram), 
            skipgram,
            args
            ).train()
 

    def _plot_losses(self, args, losses, etas):
        '''
        Display evolution of loss
        
        Parameters:
            args       Command line arguments
            losses     A list of training losses
            etas       Step sizes
        '''
        fig = figure(figsize=(12,12))
        ax1 = fig.add_subplot(1,1,1)
        plot1 = ax1.plot(losses,c='xkcd:blue',label='Loss')
        ax1.set_title(f'Training: ndim={args.ndim}, batch = {args.batch}, '
                     r'$\eta=$'
                     f'{args.eta}')
        y0, y1 = ax1.get_ylim()
        ax1.set_ylim(0, y1)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        
        ax2 = ax1.twinx()
        plot2 = ax2.plot(etas,c='xkcd:red',linestyle='dashed',label=r'$\eta$')
        eta0, eta1 = ax2.get_ylim()
        ax2.set_ylim(0, eta1)        
        ax2.set_ylabel(r'$\eta$')

        plots = plot1 + plot2
        ax1.legend(plots, [l.get_label() for l in plots])
        
        fig.savefig((Path(args.figs) / args.output).with_suffix('.png'))


class TrainSkipgrams(AbstractTrainSkipgrams):
    '''
    Adjust weights of  skipgrams after 6.8.2 of Jurafsky & Martin
    '''

    def __init__(self):
        super().__init__('train')

    def _create(self, args, rng=np.random.default_rng()):
        '''
        Create new skipgram for training
        
        Parameters:
            args      Command line parameters as parsed by parse_args()
            rng       Random number generator
        '''
        return SkipGram(Examples.create((Path(args.data) / args.input[0]).with_suffix('.pkl')),
                        ndim=args.ndim,
                        rng=rng,
                        batch=args.batch)


class RestartSkipgrams(AbstractTrainSkipgrams):
    '''
    Adjust weights of  skipgrams after 6.8.2 of Jurafsky & Martin
    '''

    def __init__(self):
        super().__init__('restart')

    def _create(self, args, rng=np.random.default_rng()):
        '''
        Instantiate skipgram from file
        
        Parameters:
            args      Command line parameters as parsed by parse_args()
            rng       Random number generator
        '''
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Restarting Training')
        return SkipGram.create((Path(args.data) / args.input[0]).with_suffix('.pkl'),
                               report=lambda s: Logger.get_instance().log(f'{__file__} {Logger.get_line()} {s}'))


class BuildDistances(Command):
    '''
    Build table of scalar products between weight vectors
    '''

    def __init__(self):
        super().__init__('build')

    def _execute(self, args, rng=np.random.default_rng()):
        '''
        Build table of scalar products between weight vectors
        
        Parameters:
            args       Command line parameters as parsed by parse_args()
            rng        Random number generator
        '''

        skipgram = SkipGram.create((Path(args.data) / args.input[0]).with_suffix('.pkl'),
                                   report=lambda s: Logger.get_instance().log(f'{__file__} {Logger.get_line()} {s}'))
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Calculating products')
        skipgram.calculate_products(args.normalize)
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Calculated products')
        skipgram.save((Path(args.data) / args.output).with_suffix('.pkl'),
                      report=lambda s: Logger.get_instance().log(f'{__file__} {Logger.get_line()} {s}'),
                      description='word vectors and distances')


class WordCluster:
    '''
    This class represents a cluster generated by AgglomerativeClustering. 
    Each cluster has two immediate descendents, either words or other cluysters
    
    Attributes:
        cluster_id      Identifies 
        children        Immediate descendents that are other clusters
        distance        Distance from parent, as calculated by AgglomerativeClustering    
        words           Immediate descendents that are
        parent          Allows traversal up the tree
    '''
    next_cluster_id = 0
    
    @staticmethod
    def build(model):
        '''
        Construct a tree of word clusters from the results of AgglomerativeClustering
        
        Parameters:
            model      Results of AgglomerativeClustering
            
        Returns:
           A list containing the clusters, also linked to form a tree
           Indices to sort the list by ascending distances
           
        '''
        n_samples, = model.labels_.shape
        n_clusters,_ = model.children_.shape
        clusters = np.empty((n_clusters),dtype=WordCluster)
        for i in range(n_clusters):
            clusters[i] = WordCluster()
            clusters[i].add_child(model.children_[i,0],clusters,n_samples)
            clusters[i].add_child(model.children_[i,1],clusters,n_samples)
            clusters[i].set_distance(model.distances_[i])
        
        return clusters, np.argsort(model.distances_)
        
            
    def __init__(self):
        self.cluster_id = WordCluster.next_cluster_id
        WordCluster.next_cluster_id += 1
        self.children = []
        self.distance = 0.0
        self.words = []
        self.parent = None
              
    def set_distance(self,distance):
        '''
        Store distance in a cluster
        '''
        self.distance = distance
        
    def add_child(self,child_index,clusters,n_samples):
        '''
        Add an immediate descendnt to c word cluster, either a word or another cluster
        
        Parameters:
            child_index
            clusters
            n_samples
        '''
        if child_index < n_samples:
            self.words.append(child_index)
        else:
            child = clusters[child_index-n_samples]
            self.children.append(child)
            child.parent = self
            
    def search_up(self,child=None,steps=-1):
        '''
        Traverse path upwards to root or for a specified number of steps
        
        Parameters:
            child    Used only for recursion: default to None
            steps    If posigtive, stop after this many steps
                     if root has not been reached
        '''
        if self.parent == None or steps == 0:
            if child.cluster_id  == self.children[0].cluster_id : return self.children[1]
            if child.cluster_id  == self.children[1].cluster_id : return self.children[0]
        else:
            return self.parent.search_up(self,steps - 1)
        
    def search_down(self,words,rng=np.random.default_rng()):
        '''
        Traverse a path down to a leaf
        
        Parameters:
            words
            rng
        '''
        for word in self.words:
            words.append(word)
            
        if len(self.children) == 0:
            return words
        else:
            return self.children[rng.choice(len(self.children))].search_down(words,rng=rng)
    
class DrawDendrogram(Command):
    '''
    Build and plot dendrogram
    '''

    def __init__(self):
        super().__init__('build-tree')

    def _execute(self, args, rng=np.random.default_rng()):
        '''
        Cluster word vectors using AgglomerativeClustering
        
        Parameters:
            args    Parsed command line parameters
            rng     Random number generator
        '''
        skipgram = SkipGram.create((Path(args.data) / args.input[0]).with_suffix('.pkl'),
                                   report=lambda s: Logger.get_instance().log(f'{__file__} {Logger.get_line()} {s}'))
        distance_matrix = ClusterUsingCRP.products_to_distances(skipgram.P)
        m, _ = distance_matrix.shape
        model = AgglomerativeClustering(
            distance_threshold=0,
            n_clusters=None,
            metric='precomputed',
            linkage='complete',
            compute_distances=True
        )

        model = model.fit(distance_matrix)
        clusters,indices = WordCluster.build(model)
        vocabulary = skipgram.examples.vocabulary
        n_samples, = model.labels_.shape
        for index in rng.choice(n_samples//2,args.Niter,replace=False):
            ingroup = clusters[index]
            branch = ingroup.search_up()   
            outgroup = branch.search_down([],rng=rng)
            if len(ingroup.words) < 2: continue
            Logger.get_instance().log(f'{__file__} {Logger.get_line()} ' 
                                      f'({vocabulary[ingroup.words[0]]},'
                                      f'{vocabulary[ingroup.words[1]]}),'
                                      f'({vocabulary[outgroup[0]]},'
                                      f'{vocabulary[outgroup[1]]})'
                                      )
                       
        self._plot(model,distance_matrix,[model.distances_[i] for i in range(n_samples//2)],args)
     
    def _plot(self,model,distance_matrix,cluster_distances,args): 
        '''
        Plot dendrogram and histogtam of distances
        
        Parameters:
            model
            distance_matrix
            cluster_distances
            args
        '''
        fig = figure(figsize=(18,18))
        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,2)
        
        dendrogram(np.column_stack(
                [model.children_, model.distances_, self.count_samples(model)]
            ).astype(float),
            truncate_mode='level', 
            p=args.levels,
            ax=ax1)
        ax1.set_title(f'Top {args.levels} levels')
        
        ax2.hist(distance_matrix.flatten(),alpha=0.5,density=True,color='xkcd:blue',label='Matrix')
        ax2.hist(cluster_distances,alpha=0.5,density=True,color='xkcd:red',label='Terminal')
        ax2.legend()
        ax2.set_title('Distances')
        ax2.set_xlabel('d')
        ax2.set_ylabel('Frequency')
        
        fig.suptitle(f'{args.input[0]}')
        fig.savefig((Path(args.figs) / args.output).with_suffix('.png'))
            

    def count_samples(self,model):
        '''
        Create the counts of samples under each node
        
        Parameters:
            model
        '''
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count
            
        return counts

  
                
class ClusterUsingCRP(Command):
    '''
    Cluster word vectors using ChineseRestaurantProcess
    '''
    def __init__(self):
        super().__init__('cluster')

    def _execute(self, args, rng=np.random.default_rng()):
        '''
        Cluster word vectors using CRP
        
        Parameters:
            args    Parsed command line parameters
            rng     Random number generator
        '''
        skipgram = SkipGram.create((Path(args.data) / args.input[0]).with_suffix('.pkl'),
                                   report=lambda s: Logger.get_instance().log(f'{__file__} {Logger.get_line()} {s}'))
        d = ClusterUsingCRP.products_to_distances(skipgram.P)
        m, _ = d.shape
        chooser = DistanceDependentChooser(d, rng=rng, alpha=args.alpha, m=m)
        clusterer = ChineseRestaurantProcess(chooser=chooser)
        clusterer.build()
        for i in range(args.Niter):
            clusterer.gibbs()

    @staticmethod
    def products_to_distances(P):
        '''
        Convert scalar product of unit vectors to a distance. 
        I have encountered a small problem with square roots of 
        negative values, hence the offset
        
        Parameters:
            P         Word vectors
        '''
        d_squared = (1 - P) / 2
        offset = min(d_squared.min(), 0)
        return np.sqrt(d_squared - offset)


def parse_args(choices):

    # Establish defaults

    window = 2
    k = 2
    weight = 0.75
    alpha = 2.0
    ndim = 128
    eta = [0.1, 0.01]
 
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
    parser.add_argument('command', choices=choices, help='Selects the function that is to be executed')
    parser.add_argument('input', nargs='+', help='List of input files')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generation')
    parser.add_argument('--data', default=data, help=f'Path to data files [{data}]')
    parser.add_argument('-o', '--output', default=None, required=True, help='File name for storing results')
    parser.add_argument('--logs', default=logs, help=f'Location for storing log files [{logs}]')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plots are shown')
    parser.add_argument('--figs', default=figs, help=f'Path used to store plots [{figs}]')
    parser.add_argument('-N', '--Niter', type=int, default=Niter, help=f'Number of iterations [{Niter}]')

    examples_group = parser.add_argument_group('Examples', description='Used for examples')
    examples_group.add_argument('-w', '--window', type=int, default=window, help=f'Width of window for context [{window}]')
    examples_group.add_argument('-k', '--k', type=int, default=k, help=f'Number of negative context words for each positive [{k}]')
    examples_group.add_argument('--weight', default=weight, type=float, help=f'The exponent from equation (6.32) [{weight}]')

    training_group = parser.add_argument_group('Training', description='Used for train')
    training_group.add_argument('-d', '--ndim', type=int, default=ndim, help=f'Length of word vectors [{ndim}]')
    training_group.add_argument('--eta', default=eta, type=float, nargs='+', help=f'Training speed [{eta}]')
    training_group.add_argument('--tau', default=None, type=int, help=f'Stop reducing eta after this many steps')
    training_group.add_argument('--ratio', default=0.9, type=float, help=f'Used to reduce eta')
    training_group.add_argument('-m', '--batch', type=int, default=batch, help=f'Number of samples in a batch [{batch}]')
    training_group.add_argument('--freq', type=int, default=freq, help=f'Interval between printing training steps [{freq}]')
    training_group.add_argument('--stepsize',choices=['Adaptive','Reducing'],default='Adaptive',help='Strategy for establishing stepsize')
    
    build_group = parser.add_argument_group(title='Build', description='Used for building distances')
    build_group.add_argument('--normalize', default=False, action='store_true',
                             help='Normalize vectors before calculating products')

   
    cluster_group = parser.add_argument_group(title='Cluster', description='Used when we cluster word vectors')
    cluster_group.add_argument('--alpha', default=alpha, type=float, help=f'Scaling parameter [{alpha}]')
    
    dendrogram_group = parser.add_argument_group(title='Draw Dendrogram', description='Used when we create dendogram')
    dendrogram_group.add_argument('--levels', type=int, default=3,help='Number of levels')
 
    return SGD.adjust_args(parser.parse_args())

def main():
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)

    Command.append([
        CreateExamples(),
        TrainSkipgrams(),
        RestartSkipgrams(),
        BuildDistances(),
        ClusterUsingCRP(),
        DrawDendrogram()
    ])
    args = parse_args(Command.get_choices())
    Command.get_command(args.command).execute(args)

    if args.show:
        show()


if __name__ == '__main__':
    main()
