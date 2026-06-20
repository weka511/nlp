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

'''Skipgrams as described in Chapter 6 of Jurafsky & Martin'''

from argparse import ArgumentParser
from glob import glob
from os.path import join
from pathlib import Path
from pickle import dump, HIGHEST_PROTOCOL, load
from time import time
import numpy as np
from vocabulary import Vocabulary
from tokenizer import generate_sentences,generate_text,generate_tokens,Token

class Examples:
    '''
    A collection of positive and negative training examplkes.
    
    Attributes:
        k
        vocabulary
        positives
        negatives
        rng

    '''
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
        
    def save(self,file):
        '''
        Save Examples using pickle.
        
        Parameters:
            file     Name of file where tables will be saved
        '''
        with open(file,'wb') as out:
            dump(self, out, HIGHEST_PROTOCOL)
            print (f'Saved examples in {file.resolve()}')    
        
    def _create_positives(self,sentence_generator):
        positives = []
        for sentence in sentence_generator:
            self._add_sentence(sentence,positives)
        return np.array(positives)
        
    def _add_sentence(self,sentence,positives):
        tokens = [self.vocabulary.tokenize(word) for word in sentence if self._is_word(word)]

        for w,c in self._generate_positive(tokens):
            positives.append([w,c])
        
    '''
    Create positive examples
    '''
    def _generate_positive(self,tokens):
        for i in range(len(tokens)):
            for j in range(-self.window,self.window + 1):
                if j == 0: continue
                if i + j < 0: continue
                if i + j >= len(tokens): continue
                yield tokens[i], tokens[i+j]
                
    '''
    Create negative examples
    '''
    def _create_negatives(self):
        indices = np.argsort(self.positives[:,0])
        positives = self.positives[indices,:]
        m,_ = positives.shape
        ws,breaks = np.unique(positives[:,0],return_index=True)
        breaks = np.hstack([breaks,[m]])
        Product = np.full((m*self.k,2),-1)
 
        for i in range(len(ws)):
            pos_min = self.window*breaks[i]
            pos_max = self.window*breaks[i+1]
            Product[pos_min:pos_max,0] = ws[i]
            Product[pos_min:pos_max,1] = self.rng.choice(len(self.vocabulary),
                                                         p=self._get_p(np.unique(positives[breaks[i]:breaks[i+1],1])),
                                                         size=pos_max - pos_min)
        return Product
 
    def _get_p(self,cs_forbidden):
        p = self.vocabulary.get_counts()
        p[cs_forbidden] = 0
        return p / p.sum()    
    
    
    def _is_word(self,s):
        '''
        Verify that a string is composed solely of letters and apostophes
        
        Parameters:
            s       The string to be tested
        '''
        return s.replace(Token.Apostrophe,'').replace(Token.Apostrophe2,'').isalpha()    
    
def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['examples',
                                           'train'])
    parser.add_argument('--seed',type=int,default=None)
    
    parser.add_argument('--data', default='./data',help='Path to corpus; also used to store ngrams') 
    parser.add_argument('-o', '--output',default=None,required=True,help='File name for storing results')
    args_examples = parser.add_argument_group('Examples','Used for command=examples')
    args_examples.add_argument('--corpus', default=None, nargs='+', help='Name(s) of corpus file(s)')
    args_examples.add_argument('-w','--window', type=int,default=2)
    args_examples.add_argument('-k','--k',type=int,default=2)
    
    return parser.parse_args()

def build_examples(args):
    examples = Examples(window=args.window,k=args.k,rng=np.random.default_rng(args.seed))
    examples.build(
        generate_sentences(
            generate_tokens(
                generate_text(
                    file_names=[globbed for name in args.corpus for globbed in glob(join(args.data, name))]
                ))))
    
    examples.save((Path(args.data) / args.output).with_suffix('.pkl'))

def train(args):
    pass

def main():
    start  = time()
    args = parse_args()

    match args.command:
        case 'examples':
            build_examples(args) 
        case 'train':
            train(args)
    
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    
if __name__=='__main__':
    main()
