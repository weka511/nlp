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

'''Template for python script'''

from argparse import ArgumentParser
from glob import glob
from os.path import join
from pathlib import Path
from time import time
import numpy as np
from vocabulary import Vocabulary
from tokenizer import generate_sentences,generate_text,generate_tokens,Token

class Examples:
    def __init__(self,window=2,k=2,rng=np.random.default_rng()):
        self.window = window
        self.k = k
        self.vocabulary = Vocabulary()
        self.positives = np.zeros((0,2))
        self.negatives = np.zeros((0,2))
        self.rng = rng
        
    def build(self,sentence_generator):
        '''
        Build examples using a generator for sentences
        
        Parameters:
            sentence_generator    Used to iterate through sentences in corpus
        '''
        self.positives = self._create_positives(sentence_generator)
        self.negatives = self._create_negatives()
        z=0
        
    def _create_positives(self,sentence_generator):
        positives = []
        for sentence in sentence_generator:
            self._add_sentence(sentence,positives)
        return np.array(positives)
        
    def _add_sentence(self,sentence,positives):
        tokens = [self.vocabulary.tokenize(word) for word in sentence if self._is_word(word)]

        for w,c in self._generate_positive(tokens):
            positives.append([w,c])
        
    def _generate_positive(self,tokens):
        for i in range(len(tokens)):
            for j in range(-self.window,self.window + 1):
                if j == 0: continue
                if i + j < 0: continue
                if i + j >= len(tokens): continue
                yield tokens[i], tokens[i+j]
    
    def _create_negatives(self):
        indices = np.argsort(self.positives[:,0])
        positives = self.positives[indices,:]
        m,_ = positives.shape
        ws,breaks = np.unique(positives[:,0],return_index=True)
        breaks = np.hstack([breaks,[m]])
        Product = np.full((m*self.k,2),-1)
 
        for i in range(len(ws)):
            w = ws[i]
            pos_min = self.window*breaks[i]
            pos_max = self.window*breaks[i+1]
            Product[pos_min:pos_max,0] = w
            cs_forbidden = np.unique(positives[breaks[i]:breaks[i+1],1])
            p = self.vocabulary.get_counts()
            p[cs_forbidden] = 0
            p /= p.sum()
            c = self.rng.choice(len(self.vocabulary),p=p,size=pos_max-pos_min)
            Product[pos_min:pos_max,1] = c
            for i in range(pos_min,pos_max):
                print (self.vocabulary[Product[i,0]],self.vocabulary[Product[i,1]])
        return Product
 
        #z=0
    
    def calculate_negatives(self):
        P = self.normalize()
        self.negatives = np.full((len(self.k*self.positives),2),-1,dtype=int)
        pos = 0
        for w,_ in self.positives:
            c = self.rng.choice(len(self.vocabulary),size=self.k,replace=False,p=P)
            self.negatives[pos:pos+self.k,0] = w
            self.negatives[pos:pos+self.k,1] = c
            pos += self.k
        print (self.negatives)
    
    def normalize(self, alpha=0.75):
        '''
        Convert counts of vocabulary items to probabilities using equation (6.32) of Jurafsky & Martin

        Parameters:
            vocabulary
            alpha      Exponent used in  equation (6.32) of Jurafsky & Martin
        '''
        P = np.zeros((len(self.vocabulary)))
        for i,count in self.vocabulary.generate_counts():
            P[i] = count**alpha
        return P/P.sum()
  
    
    def _is_word(self,s):
        '''
        Verify that a string is composed solely of letters and apostophes
        
        Parameters:
            s       The string to be tested
        '''
        return s.replace(Token.Apostrophe,'').replace(Token.Apostrophe2,'').isalpha()    
    
def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-w','--window', type=int,default=2)
    parser.add_argument('-k','--k',type=int,default=2)
    parser.add_argument('--seed',type=int,default=None)
    parser.add_argument('--corpus', default=None, nargs='+', help='Name(s) of corpus file(s)')
    parser.add_argument('--data', default='./data',help='Path to corpus; also used to store ngrams')    
    return parser.parse_args()
    
def main():
    start  = time()
    args = parse_args()
    examples = Examples(window=args.window,k=args.k,rng=np.random.default_rng(args.seed))
    examples.build(
        generate_sentences(
            generate_tokens(
                generate_text(
                    file_names=[globbed for name in args.corpus for globbed in glob(join(args.data, name))]
                ))))
    examples.calculate_negatives()
    
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    
if __name__=='__main__':
    main()
