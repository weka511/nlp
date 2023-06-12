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
from collections import Counter
from unittest import main, TestCase, skip
import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal
from scipy.special import expit

class Vocabulary:
    '''
    This class is responsible for the custody of words from a corpus.
    '''
    def __init__(self):
        self.indices = dict()
        self.counter = Counter()

    def parse(self,text):
        '''
        Parse a text into a list of indices of tokens
        '''
        Result = np.zeros(len(text),dtype=np.int64)
        for i,word in enumerate(text):
            if not word in self.indices:
                self.indices[word] = len(self.indices)
            Result[i] = self.indices[word]
            self.counter.update([Result[i]])
        return Result

    def get_count(self,index):
        '''
        Determine the number of time a token appears in text
        '''
        return self.counter[index]

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
    A class that constructs tarining data from text.
    '''

    @staticmethod
    def normalize(vocabulary, alpha=0.75):
        '''
        Convert counts of vocabulary items to probabilities using equation (6.32) of Jurafsky & Martin
        '''
        Z = sum(count**alpha for _,count in vocabulary.items())
        return {item : count**alpha / Z for item,count in vocabulary.items()}

    def __init__(self, width=2, k=2):
        self.width = width
        self.k = k

    def generate_examples(self,text,tower):
        for sentence in text:
            for i in range(len(sentence)):
                for j in range(-self.width,self.width+1):
                    if j!=0 and i+j>=0 and i+j<len(sentence):
                        yield sentence[i],sentence[i+j],+1
                        for k in self.create_negatives_for1word(sentence[i],tower):
                            yield sentence[i],k,-1

    def create_negatives_for1word(self,word,tower):
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
        '''
        self.w = rng.standard_normal((m,n))
        self.c = rng.standard_normal((m,n))
        self.n = n

    def get_product(self,i_w,i_c):
        '''
        Calculate inner product of one word vector and one context

        Parameters:
            i_w    Index of wprd vector
            i_c    Index of context vector
        '''
        return np.dot(self.w[i_w,:],self.c[i_c,:])

    def load(self,name):
        '''
        Initialize weights using stored data
        '''
        with np.load(name) as data:
            self.w = data['w']
            self.c = data['c']
            _,self.n = self.w.shape


    def save(self,name):
        '''
        Save weights in an external file
        '''
        np.savez(name,w=self.w,c=self.c)

class LossCalculator:
    '''
    Calculate loss and its derivatives
    '''
    def __init__(self,model,data):
        self.model = model
        self.data = data

    def get(self,gap, n_groups):
        '''
        Calculate loss
        '''
        return sum(self.get_loss_for_data_group(gap, i) for i in range(n_groups))

    def get_loss_for_data_group(self,gap, i_data_group):
        def get_loss_neg(j):
            i_c_neg = self.data[i_data_row+j,1]
            return np.log(expit(1-self.model.get_product(i_w,i_c_neg)))
        i_data_row = gap*i_data_group
        i_w = self.data[i_data_row,0]
        i_c_pos = self.data[i_data_row,1]
        return - (np.log(expit(self.model.get_product(i_w,i_c_pos)))
                  + sum([get_loss_neg(j) for j in range(1,gap)]))

    '''
    Calculate derivatives of loss
    '''
    def get_derivatives(self,gap, i_data_group):
        i_data_row = gap*i_data_group
        i_w = self.data[i_data_row,0]
        i_c_pos = self.data[i_data_row,1]
        term1 = expit(self.model.get_product(i_w,i_c_pos))-1
        delta_c_pos = term1 * self.model.w[i_w,:]
        term2 = [expit(self.model.get_product(i_w,self.data[i_data_row+j,1])) for j in range(1,gap)]
        delta_c_neg = [t* self.model.w[i_w,:] for t in term2]
        delta_w = term1 * self.model.c[i_c_pos,:] + sum(term2[j-1]*self.model.c[self.data[i_data_row+j,1],:] for j in range(1,gap))
        return delta_c_pos,delta_c_neg,delta_w


class Optimizer(ABC):
    '''
    Class representiong method of optimizing loss
    '''
    @staticmethod
    def create(model,data,loss_calculator,m = 16,N = 2048,eta0 = 0.05,  final_ratio=0.01, tau = 512,rng=default_rng()):
        return StochasticGradientDescent(model,data,loss_calculator,m=m, N=N, eta0=eta0, final_ratio=final_ratio,tau=tau,rng=rng)

    def __init__(self,model,data,loss_calculator,rng=default_rng()):
        self.rng = rng
        self.model = model
        self.data = data
        y = data[:,2]
        indices = np.argwhere(y>0)
        self.gap = indices.item(1) - indices.item(0)
        m,n = data.shape
        self.n_groups = int(m/self.gap)
        self.loss_calculator = loss_calculator

    @abstractmethod
    def optimize(self):
        ...

class StochasticGradientDescent(Optimizer):
    '''
    Optimizer based on Stochastic Gradient
    '''
    def __init__(self,model,data,loss_calculator,m = 16,N = 2048,eta0 = 0.05,  final_ratio=0.01, tau = 512,rng=default_rng()):
        super().__init__(model,data,loss_calculator,rng=rng)
        self.m = m # minibatch
        self.N = N
        self.eta0 = eta0
        self.eta_tau = final_ratio * self.eta0
        self.tau = tau
        self.log = []

    def optimize(self):
        total_loss = self.loss_calculator.get(self.gap, self.n_groups)
        for k in range(self.N):
            if k<self.tau:
                alpha = k/self.tau
                eta = (1.0 - alpha)*self.eta0 + alpha*self.eta_tau
            self.step(eta)
            total_loss = self.loss_calculator.get(self.gap, self.n_groups)
            print (f'Iteration={k+1:5d}, eta={eta:.4f}, Loss={total_loss:.2f}')
            self.log.append(total_loss)

    def step(self,eta):
        iws = np.zeros((self.m),dtype=np.int64)
        dws = np.zeros((self.m,self.model.n))
        iwc = np.zeros((self.gap*self.m),dtype=np.int64)
        dcs = np.zeros((self.gap*self.m,self.model.n))
        for i,index_test_set in enumerate(self.create_minibatch()):
            index_start = self.gap * index_test_set
            delta_c_pos,delta_c_neg,delta_w = self.loss_calculator.get_derivatives(self.gap, index_test_set)
            iws[i] = self.data[index_start,0]
            dws[i,:] = delta_w
            for k in range(self.gap):
                iwc[self.gap*i+k] = self.data[index_start + k,1]
                dcs[self.gap*i+k,:] = delta_c_pos if k==0 else delta_c_neg[k-1]
        for i in range(self.m):
            index_w = iws[i]
            self.model.w[index_w,:] -= eta * dws[i,:]
            for j in range(self.gap):
                index_c = iwc[self.gap*i+j]
                self.model.c[index_c,:] -= eta * dws[i,:]


    def create_minibatch(self):
        return self.rng.integers(self.n_groups,size=(self.m))


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

    main()
