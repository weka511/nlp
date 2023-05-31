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

from argparse import ArgumentParser
from collections import ChainMap, Counter
from pathlib import Path
from time import time
import numpy as np
from tokenizer import read_text, extract_sentences, extract_tokens

def count_words(docnames):
    '''
    Count all words in all documents

    Parameters:
        docnames

    Returns:
       all_words   A list of all words that occur in any document
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
    Compute tf-idf scores for list of documents

    Parameters:
        docnames  A list of pathnames for the documents that are to be processed

    Returns:
        all_words
        tf_idf
    '''


    all_words, word_counts = count_words(docnames)
    m = len(all_words)
    n = len(docnames)
    tf = np.zeros((m,n))
    tf_idf = np.zeros((m,n))
    idf = np.zeros((m))

    for i,word in enumerate(all_words):
        df = 0
        for j,wc in enumerate(word_counts):
            tf[i,j] = np.log(wc[word]+1)
            if wc[word]>0:
                df += 1
        idf[i] = np.log(n/df)
        tf_idf[i,:] = tf[i,:] * idf[i]

    return all_words,tf_idf

if __name__=='__main__':
    start  = time()
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('docnames', nargs='+', help='A list of documents to be processed')
    args = parser.parse_args()
    words,tf_idf = TfIdf(docnames=args.docnames)
    for i,word in enumerate(words):
        if np.any(tf_idf[i,:]>0):
            print (word, tf_idf[i,:])
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
