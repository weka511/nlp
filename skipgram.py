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
from os.path import isfile, join
from tempfile import TemporaryDirectory
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

    def add(self,word):
        if not word in self.indices:
            self.indices[word] = len(self.indices)
        self.counter.update([self.indices[word]])

    def get_count(self,token):
        '''
        Determine the number of time a token appears in text

        Parameters:
            token      The word whose count we want
        '''
        return self.counter[token]

    def items(self):
        '''
        Iterate through all words in vocabulary, and also give the corresponding frequency for each word
        '''
        class Items:
            '''
            Used to iterate through all words in vocabulary, and also give the corresponding frequency for each word
            '''
            def __init__(self,vocabulary):
                self.index = -1
                self.vocabulary = vocabulary

            def __iter__(self):
                '''
                Return the iterator object itself
                '''
                return self

            def __next__(self):
                '''Return the next item from the iterator. '''
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
        print (f'Loaded {name}')


    def save(self,name,docnames=[]):
        '''
        Save vocabulary in an external file
        '''
        np.savez(name,indices=self.indices,counter=self.counter,docnames=docnames)

class Index2Word:
    '''
    Companion to Vocabulary: used to find word given index
    '''
    def __init__(self,vocabulary):
        self.words = [word for word,_ in sorted([(word,position) for word,position in vocabulary.indices.items()],key=lambda tup: tup[1])]

    def __getitem__(self, key):
        '''
        Find word given index

        Parameters:
            key    Index number of word from Vocabulary

        Returns:
            word corresponding to index
        '''
        return self.words[key]

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
                for sample in self.create_negatives_for1word(word,tower):
                    yield sentence[word],sample,-1               # Emit Negative example

    def generate_words_and_context(self,sentence):
        '''
        Used to iterate through word/context pairs in a single sentence
        '''
        for i in range(len(sentence)):   # for each word
            for j in range(-self.width,self.width+1):   # for each context word within window
                if j!=0 and i+j>=0 and i+j<len(sentence):
                    yield i,i+j

    def create_negatives_for1word(self,word,tower):
        '''
        Sample vocabulary to create negative examples

        Parameters:
            word    Word being matched
            tower   Used for tower sampling
        '''
        Product = []
        while len(Product)<self.k:
            j = tower.get_sample()
            if j !=word and j not in Product:
                Product.append(j)

        return Product


class Word2Vec:
    '''
    Used to store word2vec weights
    '''
    def build(self,m,n=32,rng=default_rng(),init='gaussian'):
        '''
        Initialze weights to random values

        Parameters:
             m     Number of word vectors
             n     Dimension of word vectors
             rng   Random number generator
             init  Determines the way weights will be initialized, gaussian or uniform
        '''
        match init:
            case 'gaussian':
                self.w = rng.standard_normal((m,n))    # Word vectors
                self.c = rng.standard_normal((m,n))    # Word vectors for constant
            case 'uniform':
                limit = np.sqrt(6/(m+n))   #Goodfellow et al, eq (8.23)
                self.w = rng.uniform(low=-limit,high=limit,size=(m,n))
                self.c = rng.uniform(low=-limit,high=limit,size=(m,n))

        self.n = n

    def get_product(self,i_w,i_c):
        '''
        Calculate inner product of one word vector and one context

        Parameters:
            i_w    Index of word vector
            i_c    Index of context vector
        '''
        return np.dot(self.w[i_w,:],self.c[i_c,:])

    def create_products(self,normalized=True):
        '''
        Calculate inner products of word vectors W

        Parameters:
            normalized    Inner product should be normalized
        '''
        m,_ = self.w.shape
        Product = np.full((m,m),np.nan)
        WC = self.w + self.c
        for i in range(m):
            for j in range(i,m):
                Product[i,j] = np.dot(WC[i,:],WC[j,:])
                Product[j,i] = Product[i,j]

        if normalized:
            Normalizer = np.sqrt(Product.diagonal())
            return Product/np.outer(Normalizer,Normalizer)
        else:
            return Product


    def load(self,name,report=print):
        '''
        Initialize weights using stored data

        Parameters:
            name     File with stored weights
            report   Function, used to report success
        '''
        with np.load(name) as data:
            self.w = data['w']
            self.c = data['c']
            _,self.n = self.w.shape
            width = data['width']
            k = data['k']
            paths =data['paths']
            eta = data['eta']
            total_loss = data['total_loss']
            report (f'Loaded {name} eta={eta}, loss {total_loss:.8e}, k={k}, width={width}')

    def save(self,name,width=2,k=2,paths=[],eta=np.inf,total_loss=np.inf,report=print):
        '''
        Save weights in an external file

        Parameters:
            name       File name to save weights
            width      Window width
            k          For negative examples
            paths      Files used to create training data
            eta        Step size
            total_loss Loss for final step
            report     Function, used to report success
        '''
        np.savez(name,w=self.w,c=self.c,width=width,k=k,paths=paths,eta=eta,total_loss=total_loss)
        report (f'Saved weights in {name}')

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
        return sum(self.get_loss_for_data_group(gap, i) for i in range(n_groups))/(gap*n_groups)

    def get_loss_for_data_group(self,gap, i_data_group):
        '''
        Calculate loss for one data group (positive example and accompanying negatives)

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
        self.etas = []

    def optimize(self,is_stopping=lambda :False):
        '''
        Minimize loss: this performs stochastic gradient optimization,
        and calculates learning rate to be used at each step.

        Parameters:
            is_stopping   A callback used to stop execution gracefully

        Returns:
           Stepsize and Loss from last iteration
        '''
        total_loss = self.loss_calculator.get(self.gap, self.n_groups)
        print (f'Initial Loss={total_loss:.8e}')
        oldargs = np.seterr(divide='raise', over='raise')   # Issue 23: we need to detect division by zero
        for k in range(self.N):
            if is_stopping(): break
            self.step(self.get_eta(k))
            if k%self.freq==0:
                total_loss = self.loss_calculator.get(self.gap, self.n_groups)
                print (f'Iteration={k+1:5d}, eta={self.get_eta(k):.5e}, Loss={total_loss:.8e}')
                if abs(total_loss) < np.inf:
                    self.log.append(total_loss)
                    self.etas.append(self.get_eta(k))
                    self.checkpoint()
                else:
                    np.seterr(**oldargs) # Issue 23: put error handling back the way it was
                    raise Exception('Total loss overflow')
        total_loss = self.loss_calculator.get(self.gap, self.n_groups) #32 Update in case freq doesn't divide N!
        print (f'Final Loss={total_loss:.8e}')
        np.seterr(**oldargs) # Issue 23: put error handling back the way it was
        return self.get_eta(k),total_loss

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
            return self.eta_tau

    @abstractmethod
    def step(self,eta):
        ...

    def checkpoint(self):
        '''
        Save model in checkpoint file. Keep one backup if file exists already.
        '''
        if self.checkpoint_file == None: return
        checkpoint_file = f'{self.checkpoint_file}.npz'
        if isfile(checkpoint_file):
            replace(checkpoint_file,f'{self.checkpoint_file}.npz.bak')
        self.model.save(self.checkpoint_file)


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



if __name__=='__main__':
    class TestVocabulary(TestCase):
        def test_parse(self):
            vocabulary = Vocabulary()
            assert_array_equal(np.array([0,1,2,3,4,5,0,6,7,8,9,0,2,10]),
                             vocabulary.parse(['the', 'quick', 'brown','fox', 'jumps', 'over', 'the', 'lazy', 'dog',
                              'that', 'guards', 'the', 'brown', 'cow']))
            self.assertEqual(3,vocabulary.get_count(0))   # Number of 'the's
            self.assertEqual(2,vocabulary.get_count(2))   # Number of 'brown's
            self.assertEqual(1,vocabulary.get_count(3))   # Number of 'fox's

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

        def test_generate_examples(self):
            '''
            Verify that Positive examples are veing constructed correctly
            '''

            vocabulary = Vocabulary()
            text = ['the', 'quick', 'brown','fox', 'jumps', 'over', 'the', 'lazy', 'dog',
                              'that', 'guards', 'the', 'brown', 'cow']
            indices = vocabulary.parse(text)
            assert_array_equal(np.array([0,1,2,3,4,5,0,6,7,8,9,0,2,10]),indices)
            word2vec = ExampleBuilder()
            tower = Tower(ExampleBuilder.normalize(vocabulary))
            examples = [(word,context,y) for (word,context,y) in word2vec.generate_examples([indices],tower)]
            self.assertEqual(3*(4*(len(indices)-4) + 4 + 6),len(examples))
            self.assertEqual((0,1,1),examples[0])
            self.assertEqual((0,2,1),examples[3])
            self.assertEqual((1,0,1),examples[6])
            self.assertEqual((1,2,1),examples[9])
            self.assertEqual((1,3,1),examples[12])
            self.assertEqual((2,0,1),examples[15])
            self.assertEqual((2,1,1),examples[18])
            self.assertEqual((2,3,1),examples[21])
            self.assertEqual((2,4,1),examples[24])
            self.assertEqual((10,2,1),examples[147])
            self.assertEqual((10,0,1),examples[144])

    class TestLoss(TestCase):
        '''Test for LossCalculator'''
        def setUp(self):
            self.oldargs = np.seterr(divide='raise', over='raise')

        def tearDown(self):
            np.seterr(**self.oldargs)

        def test_sigmoid(self):
            self.assertEqual(0.5,LossCalculator.sigmoid(0))
            self.assertEqual(0,LossCalculator.sigmoid(-np.inf))
            self.assertAlmostEqual(0,LossCalculator.sigmoid(-100))
            self.assertEqual(1,LossCalculator.sigmoid(np.inf))
            self.assertAlmostEqual(1,LossCalculator.sigmoid(100))

        def test_log_sigmoid(self):
            self.assertEqual(-0.6931471805599453,LossCalculator.log_sigmoid(0))
            self.assertEqual(0,LossCalculator.log_sigmoid(1000000))
            self.assertEqual(-1000000,LossCalculator.log_sigmoid(-1000000))

        def test_save_load(self):
            '''
            Verify that loss is calculated consistency following load and save (investigation of issue #32)
            '''

            with TemporaryDirectory() as tmpdirname:
                Data = np.array([[0,1,1],
                                 [0,1015,-1],
                                 [0,1471,-1],
                                 [0,959,-1],
                                 [0,1573,-1],
                                 [0,346,-1],
                                 [0,2,1],
                                 [0,4307,-1],
                                 [0,883,-1],
                                 [0,5087,-1],
                                 [0,434,-1],
                                 [0,63,-1],
                                 [1,0,1],
                                 [1,439,-1],
                                 [1,444,-1],
                                 [1,149,-1],
                                 [1,5117,-1],
                                 [1,2018,-1]])
                word2Vec1 = Word2Vec()
                word2Vec1.build(5118)
                calculator1 = LossCalculator(word2Vec1,Data)
                loss1 = calculator1.get(6,3)
                word2Vec1.save(join(tmpdirname,'test_save_load'),report=lambda x:None)

                word2Vec2 = Word2Vec()
                word2Vec2.load(join(tmpdirname,'test_save_load.npz'),report=lambda x:None)
                calculator2 = LossCalculator(word2Vec2,Data)
                loss2 = calculator2.get(6,3)

                self.assertEqual(loss1,loss2)


    class TestWord2Vec(TestCase):
        def test_save_load(self):
            '''
            Verify that weights can be loaded and saved correctly (investigation of issue #32)
            '''
            log = []
            def report(x):
                log.append(x)

            with TemporaryDirectory() as tmpdirname:
                word2Vec1 = Word2Vec()
                word2Vec1.build(2,2)
                word2Vec1.save(join(tmpdirname,'test_save_load'),report=report)
                self.assertTrue(log[-1].startswith('Saved weights'))
                word2Vec2 = Word2Vec()
                word2Vec2.load(join(tmpdirname,'test_save_load.npz'),report=report)
                self.assertTrue(log[-1].startswith('Loaded'))
                assert_array_equal(word2Vec1.w,word2Vec2.w)
                assert_array_equal(word2Vec1.c,word2Vec2.c)

    main()
