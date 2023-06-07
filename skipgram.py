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

'''Template for python unit tests'''

from collections import Counter
from time import time
from unittest import main, TestCase, skip
import numpy as np
from numpy.random import default_rng


class Tower:
    '''
    This class supports tower sampling from a vocabulary
    '''
    def __init__(self,vocabulary,rng=default_rng()):
        self.cumulative_probabilities = np.zeros(len(vocabulary)+1)
        self.Words = []
        Z = 0
        for i,(word,freq) in enumerate(vocabulary.items()):
            self.cumulative_probabilities[i] = Z
            Z += freq
            self.Words.append(word)
        self.cumulative_probabilities[-1] = Z
        self.rng = rng

    def get_sample(self):
        return max(0,
                   np.searchsorted(self.cumulative_probabilities,
                                   self.rng.uniform())-1)

class Word2Vec:
    def __init__(self, width=2, k=2, rng = default_rng()):
        self.width = width
        self.k = k
        self.rng = rng

    def generate_positive_examples(self,text):
        for sentence in text:
            for i in range(len(sentence)):
                for j in range(-self.width,self.width+1):
                    if j!=0 and i+j>=0 and i+j<len(sentence):
                        yield sentence[i],sentence[i+j]


    def create_negatives_for1word(self,word,tower):
        Product = []
        while len(Product)<self.k:
            j = tower.get_sample()
            if j not in Product:
                Product.append(j)

        return Product

    def create_negative_examples(self,vocabulary):
        tower = Tower(vocabulary,rng=self.rng)
        for word in vocabulary:
            for index in self.create_negatives_for1word(word, tower):
                yield word,tower.Words[index]


    def build(self,vocabulary,text):
        n = len(vocabulary)
        w = self.rng.standard_normal(n)
        c = self.rng.standard_normal(n)
        # train w & c
        return w,c

def normalize(vocabulary):
    Z = sum(count for _,count in vocabulary.items())
    return {item : count/Z for item,count in vocabulary.items()}

if __name__=='__main__':
    class TestTower(TestCase):
        def uniform(self):
            return self.sample

        def test_tower(self):
            vocabulary = {'a':10,
                          'tablespoon':1,
                          'of':10,
                          'apricot' :1,
                          'jam' :2}
            tower = Tower(normalize(vocabulary),self)

            self.sample = 0
            self.assertEqual(0,tower.get_sample())
            self.sample = 0.41666666
            self.assertEqual(0,tower.get_sample())
            self.sample = 0.41666667
            self.assertEqual(1,tower.get_sample())
            self.sample = 0.91666667
            self.assertEqual(4,tower.get_sample())
            self.sample = 0.91666666
            self.assertEqual(3,tower.get_sample())
            self.sample = 1
            self.assertEqual(4,tower.get_sample())

    class TestSkipGram(TestCase):
        def test_count_words(self):
            '''
            Verify that ...
            '''
            text = [['a', 'tablespoon', 'of', 'apricot', 'jam']]
            vocabulary = {'a':10,
                          'tablespoon':1,
                          'of':10,
                          'apricot' :1,
                          'jam' :2,
                          'aardvark' : 5,
                          'my' : 5,
                          'where': 5,
                          'coaxial' : 5,
                          'seven' : 5,
                          'forever: 5'
                          'dear' : 5,
                          'if' : 5}
            word2vec = Word2Vec()
            print ('Positive examples')
            for a,b in word2vec.generate_positive_examples(text):
                print (a,b)
            print ('Negative examples')
            for a,b in word2vec.create_negative_examples(normalize(vocabulary)):
                print (a,b)

    main()
