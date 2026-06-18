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
   Exercise 3.10 from Jurafsky and Martin
   
   Use ngrams to generate random sequences
'''

from argparse import ArgumentParser
from pathlib import Path
from sys import float_info
from time import time
import numpy as np
from ngram import Ngram
from vocabulary import Vocabulary

class SentenceFactory:
    '''
    This class creates sentences from the ngram table
    '''
    def __init__(self,ngrams,m=25,rng = np.random.default_rng(),epsilon=float_info.min):
        self.ngrams = ngrams
        self.m = m
        self.rng = rng
        self.epsilon = epsilon
        self.n = ngrams.n
    
    def create(self):
        '''
        Create one sentence using ngram table
        
        Parameters:
            ngrams    Statistics for ngrams
            m         Maximum length of each sentence
            rng       Random number generator
            epsilon   Used to avoid zero probabilities
        '''
        prefix = [-1] * (self.n - 1)
        sentence = []
        for i in range(self.m):
            P = self.ngrams.get_probabilities(prefix=tuple(prefix),epsilon=self.epsilon)
            next_token = self.rng.choice(len(P),p=P)
            if next_token == -1: break
            prefix = prefix[1:] + [next_token]
            sentence.append(self.ngrams.get_word(next_token))
        return ' '.join(sentence)

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('ngrams')
    parser.add_argument('--data', default='./data',help='Path to ngram data')
    parser.add_argument('--seed',default=None,type=int,help='Used to make random number generator reproducible')
    parser.add_argument('--m',default=25,type=int,help='Maximum length of each sentence')
    parser.add_argument('--N',default=25,type=int,help='Number of sentences to generate')
    parser.add_argument('--epsilon',default=float_info.min,type=float,help='Used to avoid zero probabilities')
    return parser.parse_args()
    
def main():
    start  = time()
    args = parse_args()
    rng = np.random.default_rng(seed=args.seed)
    ngrams = Ngram.create((Path(args.data) / args.ngrams).with_suffix('.pkl'))
    sentence = SentenceFactory(ngrams,m=args.m,rng=rng,epsilon=args.epsilon)
    for i in range(args.N):
        print (sentence.create())
               
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    
if __name__=='__main__':
    main()
