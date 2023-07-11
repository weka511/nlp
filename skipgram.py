#!/usr/bin/env python

#   Copyright (C) 2023 Simon Crase

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

from abc import ABC, abstractmethod
from builtins import FloatingPointError
from collections import Counter
from os import replace
from os.path import isfile
from unittest import main, TestCase, skip
import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal, assert_array_less

class Vocabulary:
    '''
    This class is responsible for the custody of words from a corpus.
    '''
    def __init__(self):
        self.indices = dict()
        self.counter = Counter()

    def __len__(self):
        '''
        Get number of words in vocabulary
        '''
        return len(self.indices)

    def parse(self,text,verbose=False):
        '''
        Parse a text into a list of indices of tokens
        '''
        def is_word(word):
            if word.isalpha(): return True
            if len(word) > 2 and "'" in word:  # not the apostrophe from keyboard: pasted from text
                return True
            return False

        Result = np.zeros(len(text),dtype=np.int64)
        i = 0
        for word in text:
            if is_word(word):
                if not word in self.indices:
                    self.indices[word] = len(self.indices)
                Result[i] = self.indices[word]
                self.counter.update([Result[i]])
                i += 1
            else:
                if verbose:
                    print(f'Skipping {word}')
        return Result

    def get_count(self,token):
        '''
        Determine the number of time a token appears in text

        Parameters:
            token
        '''
        return self.counter[token]

    def items(self):
        '''
        Used to iterate through all words in vocabulary, and also give the corresponding frequency for each word
        '''
        class Items:
            def __init__(self,vocabulary):
                self.index = -1
                self.vocabulary = vocabulary

            def __iter__(self):
                return self

            def __next__(self):
                self.index += 1
                if self.index < len(self.vocabulary.counter):
                    return self.index,self.vocabulary.counter[self.index]
                else:
                    raise StopIteration
        return Items(self)

    def load(self,name):
        '''
        Initialize vocabulary from stored data
        '''
        with np.load(name,allow_pickle=True) as data:
            self.indices = data['indices'].item()
            self.counter = data['counter'].item()


    def save(self,name):
        '''
        Save vocabulary in an external file
        '''
        np.savez(name,indices=self.indices,counter=self.counter)

class Index2Word:
    '''
    Companion to Vocabulary: used to find word given index
    '''
    def __init__(self,vocabulary):
        self.words = [word for word,_ in sorted([(word,position) for word,position in vocabulary.indices.items()],key=lambda tup: tup[1])]

    def get_word(self,index):
        '''
        Find word given index

        Parameters:
            index    Index number of word fromm Vocabulary

        Returns:
            word corresponding to index
        '''
        return self.words[index]

class Tower:
    '''
    This class supports tower sampling from a vocabulary
    '''
    def __init__(self,probabilities,rng=default_rng()):
        self.cumulative_probabilities = np.zeros(len(probabilities)+1)
        self.Words = []
        Z = 0
        for i,(word,freq) in enumerate(probabilities.items()):
            self.cumulative_probabilities[i] = Z
            Z += freq
            self.Words.append(word)
        self.cumulative_probabilities[-1] = Z
        self.rng = rng

    def get_sample(self):
        '''
        Retrieve a random sample, respecting probabilities
        '''
        return max(0,
                   np.searchsorted(self.cumulative_probabilities,
                                   self.rng.uniform())-1)

class ExampleBuilder:
    '''
    A class that constructs training data from text.
    '''

    @staticmethod
    def normalize(vocabulary, alpha=0.75):
        '''
        Convert counts of vocabulary items to probabilities using equation (6.32) of Jurafsky & Martin

        Parameters:
            vocabulary
            alpha      Exponent used in  equation (6.32) of Jurafsky & Martin
        '''
        Z = sum(count**alpha for _,count in vocabulary.items())
        return {item : count**alpha / Z for item,count in vocabulary.items()}

    def __init__(self, width=2, k=2):
        self.width = width
        self.k = k

    def generate_examples(self,text,tower):
        '''
        Create both positive and negative examples for all words
        '''
        for sentence in text:
            for word,context in self.generate_words_and_context(sentence):
                yield sentence[word],sentence[context],+1         # Emit Positive example
                for sample in self.create_negatives_for1word(tower):
                    yield sentence[word],sample,-1               # Emit Negative example

    def generate_words_and_context(self,sentence):
        '''
        Used to iterate through word/context pairs in a single sentence
        '''
        for i in range(len(sentence)):   # for each word
            for j in range(-self.width,self.width+1):   # for each context word within window
                if j!=0 and i+j>=0 and i+j<len(sentence):
                    yield i,i+j

    def create_negatives_for1word(self,tower):
        '''
        Sample vocabulary to create negative examples
        '''
        Product = []
        while len(Product)<self.k:
            j = tower.get_sample()
            if j not in Product:
                Product.append(j)

        return Product


class Word2Vec:
    '''
    Used to store word2vec weights
    '''
    def build(self,m,n=32,rng=default_rng()):
        '''
        Initialze weights to random values

        Parameters:
             m     Number of word vectors
             n     Dimension of word vectors
             rng   Random number generator
        '''
        self.w = rng.standard_normal((m,n))    # word vectors
        self.c = rng.standard_normal((m,n))    # Word vectors for constant
        self.n = n
        assert_array_less(self.w,np.inf)    # Issue #23
        assert_array_less(self.c,np.inf)    # Issue #23

    def get_product(self,i_w,i_c):
        '''
        Calculate inner product of one word vector and one context

        Parameters:
            i_w    Index of word vector
            i_c    Index of context vector
        '''
        return np.dot(self.w[i_w,:],self.c[i_c,:])

    def create_productsW(self,normalized=True):
        '''
        Calculate inner products of vectors
        '''
        m,_ = self.w.shape
        Product = np.full((m,m),np.nan)
        for i in range(m):
            for j in range(i,m):
                Product[i,j] = np.dot(self.w[i,:],self.w[j,:])
                Product[j,i] = Product[i,j]

        if normalized:
            Normalizer = np.sqrt(Product.diagonal())
            return Product/np.outer(Normalizer,Normalizer)
        else:
            return Product

    def create_productsWC(self):
        '''
        Calculate inner products of vectors
        '''
        m,_ = self.w.shape
        Product = np.full((m,m),np.nan)
        for i in range(m):
            for j in range(m):
                Product[i,j] = np.dot(self.w[i,:],self.c[j,:])

        return Product

    def load(self,name):
        '''
        Initialize weights using stored data
        '''
        with np.load(name) as data:
            self.w = data['w']
            self.c = data['c']
            _,self.n = self.w.shape


    def save(self,name,width=2,k=2,paths=[]):
        '''
        Save weights in an external file
        '''
        np.savez(name,w=self.w,c=self.c,width=width,k=k,paths=paths)



class LossCalculator:
    '''
    Calculate loss and its derivatives
    '''
    @staticmethod
    def sigmoid(x):
        '''
        Calculate logistic function

        Parameters:
            x    Value to be squashed by logistic function

        Returns:
            Result of equation (5.4) in Jurafsky and Martin
        '''
        return 1.0/(1.0 + np.exp(-x))

    @staticmethod
    def log_sigmoid(x):
        '''
        Used to calculate log of sigmoid - attempt to fix  #23

        log sigmoid(x) = log (1/(1+exp(-x)))
                       = log(1) - log(1 + exp(-x))
                       =  - log(1 + exp(-x))

        NB: if -x is large enough to cause an overflow,
            -np.log(1+np.exp(-x)) is close to -np.log(np.exp(-x)) = -(-x)) = x
        '''
        try:
            value = -np.log(1.0 + np.exp(-x))
            return value if value > -np.inf else x
        except FloatingPointError:
            return x

    def __init__(self,model,data):
        '''
        Initialize loss calculator

        Parameters:
            model
            data
        '''
        self.model = model
        self.data = data

    def get(self,gap, n_groups):
        '''
        Calculate loss

        Parameters:
             gap        Gap between positive examples
             n_groups   Number of positive examples (each with bevy of negatives)
        '''
        return sum(self.get_loss_for_data_group(gap, i) for i in range(n_groups))

    def get_loss_for_data_group(self,gap, i_data_group):
        '''
        Calculate loss for one data group (positive example and accompanying nagatives)

        Parameters:
            gap          Gap between positive examples
            i_data_group Identifies group
        '''
        def get_loss_neg(j):
            '''
            Contribution to loss from one negative example
            '''
            i_c_neg = self.data[i_data_row+j,1]

            return LossCalculator.log_sigmoid((-self.model.get_product(i_w,i_c_neg))) #23 removed 1 -  from within sigmoid

        i_data_row = gap*i_data_group
        i_w = self.data[i_data_row,0]
        i_c_pos = self.data[i_data_row,1]
        return - (LossCalculator.log_sigmoid(self.model.get_product(i_w,i_c_pos)) + sum([get_loss_neg(j) for j in range(1,gap)]))



    def get_derivatives(self,gap, i_data_group):
        '''
        Calculate derivatives of loss

            Parameters:
                gap        Gap between positive examples
                n_groups   Number of positive examples (each with bevy of negatives)

        '''
        i_data_row = gap*i_data_group
        i_w = self.data[i_data_row,0]
        i_c_pos = self.data[i_data_row,1]
        term1 = LossCalculator.sigmoid(self.model.get_product(i_w,i_c_pos)) - 1
        delta_c_pos = term1 * self.model.w[i_w,:]
        term2 = [LossCalculator.sigmoid(self.model.get_product(i_w,self.data[i_data_row+j,1])) for j in range(1,gap)]
        delta_c_neg = [t* self.model.w[i_w,:] for t in term2]
        delta_w = term1 * self.model.c[i_c_pos,:] + sum(term2[j-1]*self.model.c[self.data[i_data_row+j,1],:] for j in range(1,gap))
        return delta_c_pos,delta_c_neg,delta_w


class Optimizer(ABC):
    '''
    Class representiong method of optimizing loss
    '''
    @staticmethod
    def create(model,data,loss_calculator,m = 16,N = 2048,eta = 0.05,  final_ratio=0.01, tau = 512,rng=default_rng(),
                 checkpoint_file = 'checkpoint', freq = 25):
        return StochasticGradientDescent(model,data,loss_calculator, checkpoint_file = checkpoint_file,
                                         freq = freq, N=N, eta=eta, final_ratio=final_ratio, tau=tau, m=m, rng=rng)

    def __init__(self,model,data,loss_calculator,checkpoint_file = 'checkpoint',
                 freq = 25,N = 2048, eta = 0.05,final_ratio=0.01, tau = 512):
        '''
        Parameters:
            model             Model being trained
            data              Training data
            loss_calculator   Used to calculate errors during training
            N                 Number of iterations
            eta              Starting value for learning rate
            final_ratio       Final learning rate as a fraction of the first
            tau               Number of steps to decrease learning rate
            checkpoint_file   Name of file to save checkpoint
            freq              Report progress and save checkpoint every FREQ iteration
        '''
        self.model = model
        self.data = data
        self.loss_calculator = loss_calculator
        self.checkpoint_file = checkpoint_file
        self.freq = freq
        self.N = N
        self.eta = eta
        self.eta_tau = final_ratio * self.eta
        self.tau = tau
        y = data[:,2]                                                  # target values for training
        indices_positive = np.argwhere(y>0)                            # Postions of positive examples
        self.gap = indices_positive.item(1) - indices_positive.item(0) # gap between positive examples
        m,n = data.shape
        self.n_groups = int(m/self.gap)          # Number of positive examples (each with bevy of negatives)
        print (f'There are {int(m/self.gap)} groups.')
        self.log = []    # Used to record losses

    def optimize(self):
        '''
        Optimize loss: this performs stochastic gradient optimization,
        and calculates learning rate to be used at each step.
        '''
        oldargs = np.seterr(divide='raise', over='raise')   # Issue 23: we need to detect division by zero
        for k in range(self.N):
            self.step(self.get_eta(k))
            if k%self.freq==0:
                total_loss = self.loss_calculator.get(self.gap, self.n_groups)
                print (f'Iteration={k+1:5d}, eta={self.get_eta(k):.4f}, Loss={total_loss:.8e}')
                if abs(total_loss) < np.inf:
                    self.log.append(total_loss)
                    self.checkpoint()
                else:
                    np.seterr(**oldargs) # Issue 23: put error handling back the way it was
                    raise Exception('Total loss overflow')

        np.seterr(**oldargs) # Issue 23: put error handling back the way it was

    def get_eta(self,k):
        '''
        Used to steadily reduce step-size eta until it reaches minimum

        Parameters:
            k Skip during optimization
        '''
        if k<self.tau:
            alpha = k/self.tau
            return (1.0 - alpha)*self.eta + alpha*self.eta_tau
        else:
            return self.eta

    @abstractmethod
    def step(self,eta):
        ...

class StochasticGradientDescent(Optimizer):
    '''
    Optimizer based on Stochastic Gradient
    '''
    def __init__(self,model,data,loss_calculator, checkpoint_file = 'checkpoint', freq = 25,  N = 2048, eta = 0.05,
                 final_ratio=0.01, tau = 512, m = 16, rng=default_rng()):
        '''
        Parameters:
            model             Model being trained
            data              Training data
            loss_calculator   Used to calculate errors during training
            m                 minibatch size
            N                 Number of iterations
            eta               Starting value for learning rate
            final_ratio       Final learning rate as a fraction of the first
            tau               Number of steps to decrease learning rate
            checkpoint_file   Name of file to save checkpoint
            freq              Report progress and save checkpoint every FREQ iteration
            rng               Random number generator
        '''
        super().__init__(model,data,loss_calculator, checkpoint_file=checkpoint_file,freq=freq,
                         N=N,eta=eta,final_ratio=final_ratio,tau=tau)
        self.m = m
        self.rng = rng

    def step(self,eta):
        self.update_weights( *self.calculate_gradients(), eta)

    def calculate_gradients(self):
        '''
        Calculate gradients so we can take one step.

        Returns:
             iws   Indices of the words that have been selected by minibatch
             dws   derivatives by dw (words)
             iwc   Indices of the contexts that have been selected by minibatch
             dcs   derivatives by dc (contexts)
        '''
        iws = np.zeros((self.m),dtype=np.int64)
        dws = np.zeros((self.m,self.model.n))      # derivatives by dw
        iwc = np.zeros((self.gap*self.m),dtype=np.int64)
        dcs = np.zeros((self.gap*self.m,self.model.n))   # derivatives by dc

        for i,index_test_set in enumerate(self.create_minibatch()):
            index_start = self.gap * index_test_set
            delta_c_pos,delta_c_neg,delta_w = self.loss_calculator.get_derivatives(self.gap, index_test_set)
            iws[i] = self.data[index_start,0]
            dws[i,:] = delta_w
            for k in range(self.gap):
                iwc[self.gap*i + k] = self.data[index_start + k,1]
                dcs[self.gap*i + k,:] = delta_c_pos if k==0 else delta_c_neg[k-1]

        return iws,dws,iwc,dcs

    def update_weights(self,iws,dws,iwc,dcs,eta):
        '''
        Update weights using calculated derivatives

        Parameters:
            iws   Indices of the words that have been selected by minibatch
            dws   derivatives by dw (words)
            iwc   Indices of the contexts that have been selected by minibatch
            dcs   derivatives by dc (contexts)
            eta   Step size
        '''
        assert_array_less(dws,np.inf)    # Issue #23
        assert_array_less(iws,np.inf)    # Issue #23
        assert_array_less(eta,np.inf)
        for i in range(self.m):
            index_w = iws[i]
            self.model.w[index_w,:] -= (eta * dws[i,:])
            for j in range(self.gap):
                index_c = iwc[self.gap*i+j]
                self.model.c[index_c,:] -= (eta * dcs[i,:])

    def create_minibatch(self):
        '''
        Sample training data to produce minibatch
        '''
        return self.rng.integers(self.n_groups,size=(self.m))

    def checkpoint(self):
        '''
        Save model in checkpoint file. Keep one backup if file exists already.
        '''
        checkpoint_file = f'{self.checkpoint_file}.npz'
        if isfile(checkpoint_file):
            replace(checkpoint_file,f'{self.checkpoint_file}.npz.bak')
        self.model.save(self.checkpoint_file)


if __name__=='__main__':
    class TestVocabulary(TestCase):
        def test_parse(self):
            vocabulary = Vocabulary()
            assert_array_equal(np.array([0,1,2,3,4,5,0,6,7]),
                             vocabulary.parse(['the', 'quick', 'brown','fox', 'jumps', 'over', 'the', 'lazy', 'dog']))
            self.assertEqual(2,vocabulary.get_count(0))

    class TestTower(TestCase):
        def uniform(self):
            return self.sample

        def test_tower(self):
            vocabulary = Vocabulary()
            vocabulary.parse(['the', 'quick', 'brown','fox', 'jumps', 'over', 'the', 'lazy', 'dog',
                              'that', 'guards', 'the', 'brown', 'cow'])

            tower = Tower(ExampleBuilder.normalize(vocabulary,alpha=1),self)

            self.sample = 0
            self.assertEqual(0,tower.get_sample())
            self.sample = 0.214
            self.assertEqual(0,tower.get_sample())
            self.sample = 0.215
            self.assertEqual(1,tower.get_sample())
            self.sample = 0.214+ 0.071
            self.assertEqual(1,tower.get_sample())
            self.sample = 0.215+ 0.071
            self.assertEqual(2,tower.get_sample())

            self.sample = 1
            self.assertEqual(11,tower.get_sample())



    class TestSkipGram(TestCase):
        def test_normalize(self):
            '''
            Verify that probabilities of vocabulary items satisfy equations (6.32,6.33) of Jurafsky & Martin
            '''
            probabilities = {'a':0.99, 'b':0.01}
            normalized_vocabulary = ExampleBuilder.normalize(probabilities)
            self.assertAlmostEqual(0.97,normalized_vocabulary['a'],places=2)
            self.assertAlmostEqual(0.03,normalized_vocabulary['b'],places=2)

        def test_count_words(self):
            '''
            Verify that ...
            '''

            vocabulary = Vocabulary()
            text = ['the', 'quick', 'brown','fox', 'jumps', 'over', 'the', 'lazy', 'dog',
                              'that', 'guards', 'the', 'brown', 'cow']
            indices = vocabulary.parse(text)
            word2vec = ExampleBuilder()
            tower = Tower(ExampleBuilder.normalize(vocabulary))
            for word,context,y in word2vec.generate_examples([indices],tower):
                print (word,context,y)

    class TestLoss(TestCase):
        '''Test for LossCalculator'''
        def setup(self):
            self.calculator = LossCalculator()

        def test_sigmoid(self):
            self.assertEqual(0.5,LossCalculator.sigmoid(0))
            self.assertEqual(0,LossCalculator.sigmoid(-np.inf))
            self.assertEqual(0,LossCalculator.sigmoid(-1000000))
            self.assertEqual(1,LossCalculator.sigmoid(np.inf))
            self.assertEqual(1,LossCalculator.sigmoid(1000000))

        def test_log_sigmoid(self):
            self.assertEqual(-0.6931471805599453,LossCalculator.log_sigmoid(0))
            self.assertEqual(0,LossCalculator.log_sigmoid(1000000))
            self.assertEqual(-1000000,LossCalculator.log_sigmoid(-1000000))

    main()
