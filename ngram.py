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
from os.path import join
from time import time
import numpy as np
from tokenizer import generate_sentences, generate_text, generate_tokens,Token

class Ngram:
    def __init__(self,n):
        self.n = n
        self.vocabulary = {}
        self.tuples = {}
        self.symbols = []
        
    def build(self,sentence):
        for t in self.generate_tuples([-1]*(self.n-1) +
                                      [self.add_word(token) for token in sentence if self.is_word(token)] +
                                      [-1]*(self.n-1)):
            try:
                self.tuples[t] += 1
            except KeyError:
                self.tuples[t] = 1
        
    def generate_tuples(self,tokens):
        for i in range(len(tokens)-self.n + 1):
            yield tuple(tokens[i:i+self.n])

    def is_word(self,token):
        return token.replace(Token.Apostrophe,'').isalpha()
    
    def add_word(self,token):
        try:
            return self.vocabulary[token]
        except KeyError:
            self.symbols.append(token)
            self.vocabulary[token] = len(self.vocabulary)
            return self.vocabulary[token]

    def get_ngram(self,key):
        return [self.symbols[i] for i in key if i > -1]
        
def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default=None, nargs='+', help='Name(s) of corpus file(s)')
    parser.add_argument('--data', default='./data')
    parser.add_argument('-n', '--n',default=3, type=int)
    return parser.parse_args()
    
def main():
    start  = time()
    args = parse_args()
    ngram = Ngram(args.n)
    file_names = [globbed for name in args.corpus for globbed in glob(join(args.data, name))]
    for sentence in generate_sentences(generate_tokens(generate_text(file_names=file_names))):
        ngram.build(sentence)
    for key,value in ngram.tuples.items():
        if value > 1:
            print (ngram.get_ngram(key),value)
            
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    
if __name__=='__main__':
    main()
