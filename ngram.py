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

'''
    Exercise 3.8 from Jurafsky and Martin
    
    Build ngram table from corpus
'''

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
        n            Length of n-grams
        vocabulary   Maps text string to tokens
        symbols      Maps tokens to text strings
        tuples       Counts for tuples
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
        
    def build(self,sentence_generator):
        '''
        Build ngrams using a generator for sentences
        
        Parameters:
            sentence_generator
        '''
        for sentence in sentence_generator:
            self._add_sentence(sentence)
            
    def save(self,file):
        '''
        Save Ngram using pickle.
        
        Parameters:
            file     Name of file where tables will be saved
        '''
        with open(file,'wb') as out:
            dump(self, out, HIGHEST_PROTOCOL)
            print (f'Saved ngrams to {file.resolve()}')
    
    def get_frequencies(self,min_count=0):
        '''
        Determine frequencies of ngrams
        
        Parameters:
            min_count
        '''
        counts = []
        for key,count in self.tuples.items():
            ngram = self._get_ngram(key)
            if self.n == len(ngram) and count >= min_count:
                counts.append(count)

        return counts
    
    def get_description(self):
        '''
        Used to describe n-grams for display
        '''
        match self.n:
            case 1:
                return 'word'
            case 2:
                return 'bigram'
            case 3:
                return 'trigram'
            case _:
                return f'{self.n}-gram'
            
    def get_probabilities(self,prefix=(-1,-1),epsilon=1):
        '''
        Determine probabilities of each token given the prefix
        
        Parameters:
            prefix
            epsilon
        '''
        P = np.full((len(self.vocabulary)),epsilon,dtype=float)
        for ngram,count in self.tuples.items():
            if ngram[:-1] == prefix:
                token = ngram[-1]
                P[token] += count
 
        return P/P.sum()
    
    def get_word(self,token):
        '''
        Look up the word that corresponds to a token
        
        Parameters:
            token
        '''
        return self.symbols[token]
    
    def _add_sentence(self,sentence):
        '''
        Grow our list of tuples using those that can be extracted from a single sentence
        
        Parameters:
            sentence   Text from one sentence
        '''
        for n_gram in self._generate_tuples( [-1]*(self.n-1) +
                                            [self._tokenize(word) for word in sentence if self._is_word(word)] +
                                            [-1]*(self.n-1)):
            try:
                self.tuples[n_gram] += 1
            except KeyError:
                self.tuples[n_gram] = 1
        
    def _generate_tuples(self,tokens):
        '''
        Extract tuples of length n from a tokenized sentence
        
        Parameters:
            tokens    Tokenized sentence padded with <start>/<end>
        '''
        for i in range(len(tokens)-self.n + 1):
            yield tuple(tokens[i:i+self.n])

    def _is_word(self,s):
        '''
        Verify that a string is composed solely of letters and apostophes
        
        Parameters:
            s       The string to be tested
        '''
        return s.replace(Token.Apostrophe,'').isalpha()
    
    def _tokenize(self,word):
        '''
        Convert a word to a token; if we haven't seen it before, create a new token
        
        Parameters:
            word     A string of characters comprising a single wird
        '''
        try:
            return self.vocabulary[word]
        except KeyError:
            self.vocabulary[word] = len(self.symbols)
            self.symbols.append(word)
            return self.vocabulary[word]

    def _get_ngram(self,tokens):
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
    parser.add_argument('-o', '--output',default=Path(__file__).stem)
    parser.add_argument('--show', default=False,action='store_true')
    parser.add_argument('--figs', default='./figs')
    return parser.parse_args()
    
def main():
    start  = time()
    args = parse_args()
    ngram = Ngram(args.n)
    file_names = [globbed for name in args.corpus for globbed in glob(join(args.data, name))]
    ngram.build(generate_sentences(generate_tokens(generate_text(file_names=file_names))))

    ngram.save((Path(args.data) / args.output).with_suffix('.pkl'))
    
    fig = figure(figsize=(10,10))
    ax1 = fig.add_subplot(2,2,1)
    ax1.hist(ngram.get_frequencies(),bins='sqrt',color='xkcd:blue',density=True)
    ax1.set_title(f'Frequencies for all {ngram.get_description()}s')
    
    ax2 = fig.add_subplot(2,2,2)
    ax2.hist(ngram.get_frequencies(min_count=2),bins='fd',color='xkcd:red',density=True)
    ax2.set_title(f'Frequencies for {ngram.get_description()}s with two occurences or more')
    
    ax3 = fig.add_subplot(2,2,3)
    ax3.hist(ngram.get_probabilities(prefix=(0,1)),bins='fd')
    fig.savefig((Path(args.figs) / args.output).with_suffix('.png'))
    
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
    
if __name__=='__main__':
    main()
