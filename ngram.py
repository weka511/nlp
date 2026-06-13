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
from matplotlib.pyplot import figure,show
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
    
    @staticmethod
    def create(file_name):
        '''
        A factory method to instantiate an Ngram from a saved file
        '''
        with open(file_name, 'rb') as inp:
            product = load(inp) 
            print (f'Loaded ngrams from {file_name.resolve()}')
            return product
                    
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
    
    def save(self,output_file):
        '''
        Save Ngram using pickle.
        
        Parameters:
            output_file
        '''
        with open(output_file,'wb') as out:
            dump(self, out, HIGHEST_PROTOCOL)
            print (f'Saved ngrams to {output_file.resolve()}')
    
    def get_frequencies(self,min_count=0):
        counts = []
        for key,count in self.tuples.items():
            ngram = self.get_ngram(key)
            if self.n == len(ngram) and count >= min_count:
                counts.append(count)

        return counts
        
def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default=None, nargs='+', help='Name(s) of corpus file(s)')
    parser.add_argument('--data', default='./data')
    parser.add_argument('-n', '--n',default=3, type=int)
    parser.add_argument('-o', '--output',default=Path(__file__).stem)
    parser.add_argument('--show', default=False,action='store_true')
    parser.add_argument('--figs', default='./figs')
    parser.add_argument('--load',default=None)
    return parser.parse_args()
    
def main():
    start  = time()
    args = parse_args()
    ngram = Ngram(args.n)
    file_names = [globbed for name in args.corpus for globbed in glob(join(args.data, name))]
    for sentence in generate_sentences(generate_tokens(generate_text(file_names=file_names))):
        ngram.add_sentence(sentence)
 
    ngram.save((Path(args.data) / args.output).with_suffix('.pkl'))
    
    fig = figure(figsize=(10,10))
    ax1 = fig.add_subplot(2,1,1)
    ax1.hist(ngram.get_frequencies(),bins=100,color='xkcd:blue',density=True)
    ax1.set_title('Frequencies for all tuples')
    ax2 = fig.add_subplot(2,1,2)
    ax2.hist(ngram.get_frequencies(min_count=2),bins='fd',color='xkcd:red',density=True)
    ax2.set_title('Frequencies for tuples with two occurences or more')
    fig.savefig((Path(args.figs) / args.output).with_suffix('.png'))
    
    if args.load != None:
        ngram1 = Ngram.create((Path(args.data) / args.load).with_suffix('.pkl'))
        for key,value in ngram1.tuples.items():
            if value > 2:
                print (ngram.get_ngram(key),value)    
           
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
    
if __name__=='__main__':
    main()
