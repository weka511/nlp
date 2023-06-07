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
from unittest import main, TestCase
import numpy as np
from numpy.random import default_rng

def create_positives(text,width=2):
    Product = []
    for sentence in text:
        for i in range(len(sentence)):
            for j in range(-width,width+1):
                if j!=0 and i+j>=0 and i+j<len(sentence):
                    Product.append((sentence[i],sentence[i+j]))
    return Product

def create_negatives(vocabulary,k=2,rng=default_rng()):

    def create_tower():
        Product = np.zeros((len(vocabulary))+1)
        Words = []
        Z = 0
        i = 0
        for word,freq in vocabulary.items():
            Product[i] = Z
            Z += freq
            i+= 1
            Words.append(word)
        Product[i] = Z
        return Product,Words

    def create_negatives_for1word(word):
        Product = []
        while len(Product)<k:
            u = rng.uniform()
            j = np.searchsorted(Tower,u,side='right')
            if j not in Product:
                Product.append(j)

        return Product

    Tower,Words = create_tower()
    Product = []
    for word in vocabulary:
        for index in create_negatives_for1word(word):
            Product.append((word,Words[index]))
    return Product

def build_skip_grams(vocabulary,text, width=2, k=2):
    n = len(vocabulary)
    rng = default_rng()
    w = rng.standard_normal(n)
    c = rng.standard_normal(n)
    positive_examples = create_positives(text,width)
    negative_examples = create_negatives(vocabulary,k,rng=rng)
    return w,c

def normalize(vocabulary):
    Z = sum(count for _,count in vocabulary.items())
    return {item : count/Z for item,count in vocabulary.items()}

if __name__=='__main__':
    class TestSummat(TestCase):
        def test_count_words(self):
            '''
            Verify that ...
            '''
            text = [['a', 'tablespoon', 'of', 'apricot', 'jam']]
            vocabulary = {'a':10,
                          'tablespoon':1,
                          'of':10,
                          'apricot' :1,
                          'jam' :2}
            w,c = build_skip_grams(normalize(vocabulary),text)
            z=0

    main()
