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

from collections import Counter
from unittest import main, TestCase, skip
import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal

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

    @staticmethod
    def normalize(vocabulary, alpha=0.75):
        '''
        Convert counts of vocabulary items to probabilities using equation (6.32) of Jurafsky & Martin
        '''
        Z = sum(count**alpha for _,count in vocabulary.items())
        return {item : count**alpha / Z for item,count in vocabulary.items()}

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

            tower = Tower(Word2Vec.normalize(vocabulary,alpha=1),self)

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
            normalized_vocabulary = Word2Vec.normalize(probabilities)
            self.assertAlmostEqual(0.97,normalized_vocabulary['a'],places=2)
            self.assertAlmostEqual(0.03,normalized_vocabulary['b'],places=2)

        def test_count_words(self):
            '''
            Verify that ...
            '''
            # text = [['a', 'tablespoon', 'of', 'apricot', 'jam']]
            # vocabulary = {'a':10,
                          # 'tablespoon':1,
                          # 'of':10,
                          # 'apricot' :1,
                          # 'jam' :2,
                          # 'aardvark' : 5,
                          # 'my' : 5,
                          # 'where': 5,
                          # 'coaxial' : 5,
                          # 'seven' : 5,
                          # 'forever': 5,
                          # 'dear' : 5,
                          # 'if' : 5}
            vocabulary = Vocabulary()
            text = ['the', 'quick', 'brown','fox', 'jumps', 'over', 'the', 'lazy', 'dog',
                              'that', 'guards', 'the', 'brown', 'cow']
            indices = vocabulary.parse(text)
            word2vec = Word2Vec()
            print ('Positive examples')
            for a,b in word2vec.generate_positive_examples([indices]):
                print (a,b)
            print ('Negative examples')
            for a,b in word2vec.create_negative_examples(Word2Vec.normalize(vocabulary)):
                print (a,b)

    main()
