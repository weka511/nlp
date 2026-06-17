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
from sys import float_info
from time import time
import numpy as np
from matplotlib.pyplot import figure,show
from matplotlib import rc
import matplotlib.patches as mpatches
from tokenizer import generate_sentences, generate_text, generate_tokens,Token

class Ngram:
    '''
    This class represents a collection of n-grams
    
    Attributes:
        n            Length of n-grams
        token        Maps text strings to tokens
        symbols      Maps tokens to text strings
        tuples       Counts for tuples
    '''

    @staticmethod
    def create(file_name):
        '''
        A factory method to instantiate an Ngram from a saved file
        
        Parameters:
            file_name    Name of file where ngrams have been stored
        '''
        with open(file_name, 'rb') as inp:
            product = load(inp) 
            print (f'Loaded ngrams from {file_name.resolve()}')
            return product
                    
    def __init__(self,n):
        '''
        Parameters:
            n       Length of n-grams
        '''
        self.n = n
        self.token = {}
        self.tuples = {}
        self.symbols = []
        
    def build(self,sentence_generator):
        '''
        Build ngrams using a generator for sentences
        
        Parameters:
            sentence_generator    Used to iterate through sentences in corpus
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
            min_count  Include only ngrams whose count is at least this value
        '''
        return [count 
                for key,count in self.tuples.items() 
                if self.n == len(self._get_ngram(key)) and count >= min_count ]
    
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
            
    def get_probabilities(self,prefix=(-1,-1),epsilon=float_info.min):
        '''
        Determine probabilities of each token given the prefix
        
        Parameters:
            prefix    A tuple that is one shorter than our ngrams. We will
                      calculate the probabilities for all ngrams that start
                      with this prefix.
            epsilon   An amount that will be added to all counts for smoothing
        '''
        P = np.full((len(self.token)),epsilon,dtype=float)
        for ngram,count in self.tuples.items():
            if ngram[:-1] == prefix:
                token = ngram[-1]
                P[token] += count

        return P/P.sum()
    
    def get_word(self,token):
        '''
        Look up the word that corresponds to a token
        
        Parameters:
            token      An index into symbol table
        '''
        return self.symbols[token]
    
    def get_branching_factors(self):
        '''
        Calculate branching factors
        
        Returns:
            Branching factors
            Word with counts, in descending order by count
        '''
        m = len(self.token)
        pairs = np.zeros((m,m),dtype=int)
        for ngram,count in self.tuples.items():
            for i in range(self.n-1):
                if ngram[i] != -1 and ngram[i+1] != -1:
                    pairs[ngram[i],ngram[i+1]] = 1
        branching_factors = pairs.sum(axis=1)
        
        indices = np.argsort(branching_factors)[::-1]
        words_with_counts = []
        for i in range(m):
            token = indices[i]
            words_with_counts.append((self.get_word(token),branching_factors[token]))
            
        return branching_factors,words_with_counts             
    
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
            return self.token[word]
        except KeyError:
            self.token[word] = len(self.symbols)
            self.symbols.append(word)
            return self.token[word]

    def _get_ngram(self,tokens):
        '''
        Convert a tuple of tokens to display form
        
        Parameters:
            tokens   Tuple to be converted
        '''
        return tuple([self.symbols[i] for i in tokens if i > -1])            
        
def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default=None, nargs='+', help='Name(s) of corpus file(s)')
    parser.add_argument('--data', default='./data',help='Path to corpus; also used to store ngrams')
    parser.add_argument('-n', '--n',default=3, type=int,help='Lenghth of ngrams')
    parser.add_argument('-o', '--output',default=Path(__file__).stem,help='File name for storing ngrams')
    parser.add_argument('--show', default=False,action='store_true',help='Controls whether plots are shown')
    parser.add_argument('--figs', default='./figs',help='Path used to store plots')
    return parser.parse_args()

    
def main():
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)

    start  = time()
    args = parse_args()
    ngram = Ngram(args.n)
    ngram.build(
        generate_sentences(
            generate_tokens(
                generate_text(
                    file_names=[globbed for name in args.corpus for globbed in glob(join(args.data, name))]
                ))))
    
    ngram.save((Path(args.data) / args.output).with_suffix('.pkl'))
    
    fig = figure(figsize=(12,12))
    fig.suptitle(f'Generating {args.n}-grams from {" ".join(args.corpus)}')
    
    ax1 = fig.add_subplot(2,2,1)
    ax1.hist(ngram.get_frequencies(),bins='sqrt',color='xkcd:blue',density=True)
    ax1.set_title(f'Frequencies for all {ngram.get_description()}s')
    
    ax2 = fig.add_subplot(2,2,2)
    ax2.hist(ngram.get_frequencies(min_count=2),bins='fd',color='xkcd:blue',density=True)
    ax2.set_title(f'Frequencies for {ngram.get_description()}s with two occurences or more')
    
    ax3 = fig.add_subplot(2,2,3)
    branching_factors,tokens_with_counts = ngram.get_branching_factors()
    ax3.plot(branching_factors,color='xkcd:blue')
    ax3.set_title(f'Branching Factors: mean={np.mean(branching_factors):.1f}')
    ax3.set_xlabel('Token')
    ax3.set_ylabel('Branching Factor')
    legend_texts = []
    m = 60
    for i in range(m):
        word,count = tokens_with_counts[i]
        legend_texts.append(f'{word} {count}')
    blank_handles = [mpatches.Patch(color='none') for _ in legend_texts]
    ax3.legend(blank_handles, legend_texts, 
               title=f'Top {m} words',handlelength=0, handletextpad=0,ncols=4)
    
    fig.tight_layout(h_pad=2)
    fig.savefig((Path(args.figs) / args.output).with_suffix('.png'))
    
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    
    if args.show:
        show()
    
if __name__=='__main__':
    main()
