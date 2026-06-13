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

'''Exercises from Chapter 3 of Jurafsky and Martin'''

from argparse import ArgumentParser
from glob import glob
from io import BytesIO
from os.path import join
from pathlib import Path
from pickle import dump, HIGHEST_PROTOCOL, load
from time import time
import numpy as np
from tokenizer import generate_sentences, generate_text, generate_tokens,Token

class Ngram:
    '''
    This class represents a collection of n-grams
    
    Attributes:
        n
        vocabulary
        symbols
        tuples
    '''
    def __init__(self,n):
        self.n = n
        self.vocabulary = {}
        self.tuples = {}
        self.symbols = []
        
    def add_sentence(self,sentence):
        '''
        Grow our list of tuples using thse that cen be extracted from sentence
        
        Parameters:
            sentence
        '''
        for n_gram in self.generate_tuples( [-1]*(self.n-1) +
                                            [self.tokenize(word) for word in sentence if self.is_word(word)] +
                                            [-1]*(self.n-1)):
            try:
                self.tuples[n_gram] += 1
            except KeyError:
                self.tuples[n_gram] = 1
        
    def generate_tuples(self,tokens):
        '''
        Extract typles of length n from a tokenized sentence
        
        Parameters:
            tokens
        '''
        for i in range(len(tokens)-self.n + 1):
            yield tuple(tokens[i:i+self.n])

    def is_word(self,token):
        '''
        Verify that a token is composed solely of letters and apostophes
        
        Parameters:
            token
        '''
        return token.replace(Token.Apostrophe,'').isalpha()
    
    def tokenize(self,word):
        '''
        Convert a word to a token; if we haven't seeit it before, create a new token
        
        Parameters:
            word
        '''
        try:
            return self.vocabulary[word]
        except KeyError:
            self.symbols.append(word)
            self.vocabulary[word] = len(self.vocabulary)
            return self.vocabulary[word]

    def get_ngram(self,tokens):
        '''
        Convert a tuple of tokens to display form
        
        Parameters:
            tokens
        '''
        return tuple([self.symbols[i] for i in tokens if i > -1])
        
def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default=None, nargs='+', help='Name(s) of corpus file(s)')
    parser.add_argument('--data', default='./data')
    parser.add_argument('-n', '--n',default=3, type=int)
    parser.add_argument('-', '--output',default=Path(__file__).stem)
    return parser.parse_args()
    
def main():
    start  = time()
    args = parse_args()
    ngram = Ngram(args.n)
    file_names = [globbed for name in args.corpus for globbed in glob(join(args.data, name))]
    for sentence in generate_sentences(generate_tokens(generate_text(file_names=file_names))):
        ngram.add_sentence(sentence)
 
    output_file = (Path(args.data) / args.output).with_suffix('.pkl')
    with open(output_file,'wb') as out:
        dump(ngram, out, HIGHEST_PROTOCOL)
        print (f'Saved ngrams to {output_file.resolve()}')

    with open(output_file, 'rb') as inp:
        ngram1 = load(inp) 
        print (f'Loaded ngrams from {output_file.resolve()}')
        for key,value in ngram1.tuples.items():
            if value > 3:
                print (ngram.get_ngram(key),value)
                
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    
if __name__=='__main__':
    main()
