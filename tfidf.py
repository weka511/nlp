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

'''Compute tf-idf scores for list of documents'''

from collections import ChainMap, Counter
from pathlib import Path
from time import time
from unittest import main, TestCase
import numpy as np
from tokenizer import read_text, extract_sentences, extract_tokens

def count_words(docnames):
    '''
    Count all words in a collection of documents

    Parameters:
        docnames

    Returns:
       vocabulary  A list of all words that occur in any document
       word_counts A list of Counters(dictionaries), one for each document, showing count
                   of each word that actually occurs in document
    '''
    dicts = ChainMap()
    word_counts = []
    for file_name in docnames:
        text = list(read_text(file_names=[file_name]))
        tokens = extract_tokens(text)
        words = [word for sentence in extract_sentences(tokens) for word in sentence if word.isalpha()]
        word_counts.append(Counter(words))
        dicts.maps.append(word_counts[-1])
    return list(dicts), word_counts

def TfIdf(docnames=[]):
    '''
    Compute tf-idf scores for list of documents. Based on
    Dan Jurafsky & James H. Martin - Speech and Language Processing

    Parameters:
        docnames  A list of pathnames for the documents that are to be processed

    Returns:
        vocabulary  The list of words in the document
        tf_idf      An array of dimension m x n containing Tf-Idf scores, where:
                       m represents the number of words in the vocabulary, excluding words that occur in all documents
                       n is the number of documents in the corpus
    '''


    vocabulary, word_counts = count_words(docnames)
    m = len(vocabulary)
    n = len(docnames)
    tf = np.zeros((m,n))
    tf_idf = np.zeros((m,n))
    idf = np.zeros((m))

    for i,word in enumerate(vocabulary):
        df = 0
        for j,wc in enumerate(word_counts):
            tf[i,j] = np.log(wc[word]+1)
            if wc[word]>0:
                df += 1
        idf[i] = np.log(n/df)
        tf_idf[i,:] = tf[i,:] * idf[i]
    selector = np.argwhere(tf_idf.any(axis=1)).flatten()
    tf_idf_reduced = np.zeros((len(selector),n))
    for i,k in enumerate(selector):
        tf_idf_reduced[i,:] =tf_idf[k,:]
    return [vocabulary[i] for i in selector],tf_idf_reduced

def create_inner_products(tf_idf):
    _,n = tf_idf.shape
    product = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            product[i,j] = np.dot(tf_idf[:,i],tf_idf[:,j]) if j>=i else product[j,i]
    return product

if __name__=='__main__':
    class TestSummat(TestCase):
        def test_count_words(self):
            '''
            Verify that vocabulary has the right words, and that the word counts are in the same sequence
            as the file names for the corpus
            '''
            vocabulary,word_counts = count_words(['nano-corpus.txt','nano-corpus1.txt'])
            self.assertIn('earl',vocabulary)
            self.assertIn('queen',vocabulary)
            wc_nano_corpus= word_counts[0]
            self.assertEqual(1,wc_nano_corpus['king'])
            self.assertEqual(0,wc_nano_corpus['earl'])
            wc_nano_corpus1= word_counts[1]
            self.assertEqual(0,wc_nano_corpus1['king'])
            self.assertEqual(1,wc_nano_corpus1['earl'])

        def test_TfIdf(self):
            vocabulary,tf_idf=TfIdf(docnames = [
                'nano-corpus.txt',
                'nano-corpus1.txt'
            ])
            self.assertEqual(5,len(vocabulary))
            self.assertIn('countess',vocabulary)

    main()
