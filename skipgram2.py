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
from scipy.special import expit
from vocabulary import Vocabulary
from tokenizer import generate_sentences,generate_text,generate_tokens,Token
from shared.utils import Logger

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
    
    @staticmethod
    def create(file_name,logger):
        '''
        A factory method to instantiate a set of Examples from a saved file
        
        Parameters:
            file_name    Name of file where examples have been stored
        '''
        with open(file_name, 'rb') as inp:
            product = load(inp) 
            logger.log(f'Loaded examples from {file_name.resolve()}')
            return product
        
    def __init__(self,window=2,k=2,rng=np.random.default_rng(),alpha=0.75):
        self.window = window
        self.k = k
        self.vocabulary = Vocabulary()
        self.positives = np.zeros((0,2))
        self.negatives = np.zeros((0,2))
        self.rng = rng
        self.alpha = alpha
        
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
        self.positives = self.positives[indices,:]
        m,_ = self.positives.shape
        ws,breaks = np.unique(self.positives[:,0],return_index=True)
        breaks = np.hstack([breaks,[m]])
        Product = np.full((m*self.k,2),-1)
        counts = self.vocabulary.get_counts()**self.alpha
        for i in range(len(ws)):
            pos_min = self.window*breaks[i]
            pos_max = self.window*breaks[i+1]
            Product[pos_min:pos_max,0] = ws[i]
            Product[pos_min:pos_max,1] = self.rng.choice(len(self.vocabulary),
                                                         p=self._get_p(
                                                             np.unique(self.positives[breaks[i]:breaks[i+1],1]),counts),
                                                         size=pos_max - pos_min)
        return Product
 
    def _get_p(self,forbidden,counts):
        '''
        Calculate probabilities using equation (6.32)
        
        Parameters:
            forbidden
            counts
                    
        '''
        p = counts.copy()
        p[forbidden] = 0
        return p / p.sum()    
    
    
    def _is_word(self,s):
        '''
        Verify that a string is composed solely of letters and apostophes
        
        Parameters:
            s       The string to be tested
        '''
        return s.replace(Token.Apostrophe,'').replace(Token.Apostrophe2,'').isalpha()    

class SkipGram:
    
    @staticmethod
    def create_probabilities(m,n,rng):
        Product = rng.uniform(size=(m,n))
        return Product/Product.sum(axis=1,keepdims=True)
    
    def __init__(self,examples,n=128,logger=None,rng=np.random.default_rng(),minibatch=1):
        self.examples = examples
        self.n = n
        self.examples = examples
        n_words = len(examples.vocabulary)
        self.w = SkipGram.create_probabilities(n_words,n,rng)
        self.c = SkipGram.create_probabilities(n_words,n,rng)
        self.rng = rng
        self.minibatch = minibatch
        
    def step(self,eta=0.01):
        n,_ = self.examples.positives.shape
        for i in self.rng.integers(n,size=(self.minibatch)):
            w_index = self.examples.positives[i,0]
            c_index = self.examples.positives[i,1]
            w = self.w[w_index,:]
            c= self.c[c_index,:]
            L_c_pos = expit(np.dot(w,c)) * w     # (6.35)
            L_c_neg = []
            for j in range(self.examples.k):
                c_neg_index = self.examples.negatives[i*self.examples.k+j,1]
                c_neg=  self.c[c_neg_index,:] 
                L_c_neg.append(expit(np.dot(w,c_neg)) * w)  # (6.36)
                
            L_w = expit(np.dot(w,c) - 1) * c # (6.37)
            for j in range(self.examples.k):
                c_neg_index = self.examples.negatives[i*self.examples.k+j,1]
                c_neg=  self.c[c_neg_index,:] 
                L_w += (expit(np.dot(w,c_neg)) * c_neg)
                
            self.c[c_index,:] -= eta * L_c_pos   
            self.w[w_index,:] -= eta * L_w
            for j in range(self.examples.k):
                c_neg_index = self.examples.negatives[i*self.examples.k+j,1]
                self.c[c_neg_index,:]  -= eta*L_c_neg[j]            
                           
        return self.get_loss()
        
    
    def get_loss(self):
        '''
        Compute loss following equation (6.34)
        '''
        n,_ = self.examples.positives.shape
        loss = 0.0
        for i in range(n):
            w_index = self.examples.positives[i,0]
            c_index = self.examples.positives[i,1]
            w = self.w[w_index,:]
            c= self.c[c_index,:]
            loss -= np.log(expit(np.dot(w,c)))
            for j in range(self.examples.k):
                assert w_index == self.examples.negatives[i*self.examples.k+j,0]
                c_neg_index = self.examples.negatives[i*self.examples.k+j,1]
                c_neg= self.c[c_neg_index,:]
                loss -= np.log(expit(-np.dot(w,c_neg)))
        return loss

        
def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['examples','train'])
    parser.add_argument('--seed',type=int,default=None)
    parser.add_argument('--data', default='./data',help='Path to corpus; also used to store ngrams') 
    parser.add_argument('-o', '--output',default=None,required=True,help='File name for storing results')
    parser.add_argument('--logs', default='./logs', help='Location for storing log files')
    
    examples_group = parser.add_argument_group('Examples','Used for command=examples')
    examples_group.add_argument('--corpus', default=None, nargs='+', help='Name(s) of corpus file(s)')
    examples_group.add_argument('-w','--window', type=int,default=2)
    examples_group.add_argument('-k','--k',type=int,default=2)
    examples_group.add_argument('--alpha',default=0.75,type=float)
    
    training_group = parser.add_argument_group('Training','Used for train command')
    training_group.add_argument('--examples',default=None)
    training_group.add_argument('-n','--n',type=int,default=128)
    training_group.add_argument('--eta',default=0.01,type=float)
    training_group.add_argument('-m','--minibatch',type=int,default=1)
    
    
    return parser.parse_args()

def build_examples(args,rng=np.random.default_rng()):
    with Logger(Path(__file__).stem,path=args.logs) as logger:
        examples = Examples(window=args.window,k=args.k,rng=rng)
        examples.build(
            generate_sentences(
                generate_tokens(
                    generate_text(
                        file_names=[globbed for name in args.corpus for globbed in glob(join(args.data, name))]
                    ))))
        
        examples.save((Path(args.data) / args.output).with_suffix('.pkl'))

def train(args,rng=np.random.default_rng()):
    with Logger(Path(__file__).stem,path=args.logs) as logger:
        trainer = SkipGram(Examples.create((Path(args.data) / args.examples).with_suffix('.pkl'),logger),
                           n=args.n,
                           logger=logger,
                           rng=rng,
                           minibatch=args.minibatch)
        for _ in range(10):
            print (trainer.step(eta=args.eta))

def main():
    start  = time()
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    match args.command:
        case 'examples':
            build_examples(args,rng=rng) 
        case 'train':
            train(args,rng=rng)
    
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    
if __name__=='__main__':
    main()
