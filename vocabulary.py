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
    This module provides support for Vocabulary,
    a mapping between words and tolkens
'''
__version__ = '1.1'
__author__ = 'Simon Crase'

from pickle import dump,load
from unittest import main, TestCase, skip
import numpy as np

class Vocabulary:
    '''
    This class maps between words and tokens
    
    Attributes:
        symbols  List of all tokens (including <SOS>/EOS>).
                 Each token will be represeneted by its index in this table
        tokens   Map text version of token to index         
        counts   Number of times each symbol appears
    '''
    @staticmethod
    def create(file_name):
        '''
        Load a vocabulary that has previously been saved
        
        Parameters:
            file_name    Path to file 
        '''
        with open(file_name, "rb") as file:
            loaded_data = load(file)
            return Vocabulary(symbols=loaded_data['symbols'],
                              token=loaded_data['token'],
                              counts=loaded_data['counts'])
        
    def __init__(self,
                 sentence_tokens : bool = False,
                 symbols = [],
                 token = {},
                 counts = []):
        '''
        Initialize vocabulary
        Parameters:
            sentence_tokens   Initialize vocabulary with start and end of sentence tolens
        '''
        if sentence_tokens:
            self.symbols = [] # This appeara redundant, but test fails 
            self.token = {}   # if I ue defaults.
            self.counts = []            
            self.SOS = self.tokenize('<SOS>')
            self.EOS = self.tokenize('<EOS>')
        else:
            self.symbols = symbols
            self.token = token
            self.counts = counts            
        
    def __len__(self):
        '''
        Get number of words in vocabulary
        '''
        return len(self.symbols)    
    
    def __getitem__(self,token):
        '''
        
        Look up the word that corresponds to a token
        
        Parameters:
            token      An index into symbol table
        '''         
        return self.get_word(token)
            
    def tokenize(self,word):
        '''
        Convert a word to a token; if we haven't seen it before, create a new token
        
        Parameters:
            word     A string of characters comprising a single wird
        '''
        try:
            token = self.token[word]
            self.counts[token] += 1
        except KeyError:
            self.token[word] = len(self.symbols)
            self.symbols.append(word)
            token = self.token[word]
            self.counts.append(1)
        return token
        
    def get_word(self,token):
        '''
        Look up the word that corresponds to a token
        
        Parameters:
            token      An index into symbol table
        '''
        return self.symbols[token]
    
    def get_count(self,token):
        '''
        Determine the number of times a token appears in text

        Parameters:
            token      The word whose count we want
        '''
        return self.counts[token]
    
    def get_counts(self):
        '''
        Retrieve counts for all tokens
        '''        
        return np.array(self.counts)
    
    def generate_counts(self):
        '''
        Iterate through all tokens, with their counts
        '''
        for token,count in enumerate(self.counts):
            yield token,count

 
    def parse(self,text):
        '''
        Parse a text into a list of indices of tokens
        
        Parameters:
            text     The text to be parsed
        '''
        def is_word(word):
            if word.isalpha(): return True
            if len(word) > 2 and "'" in word:  # not the apostrophe from keyboard: pasted from text
                return True
            return False

        Result = np.zeros(len(text),dtype=np.int64)
        i = 0
        for word in text:
            Result[i] = self.tokenize(word)
            i += 1
    
        return Result
    
    def save(self,file_name):
        '''
        Save vocabulary in a file
        
        Parameters:
            file_name    Path to file 
        '''
        with open(file_name,'wb') as out:
            dump({
                'symbols':self.symbols,
                'token':self.token,
                'counts':self.counts
                },
                 out)     
    
class TestVocabulary(TestCase):
    '''
    Test case for no sentence tokens
    '''
    def setUp(self):
        self.vocabulary = Vocabulary()
        for word in ['the', 'quick', 'brown','fox', 'jumps', 'over', 'the', 'lazy', 'dog',
                          'that', 'guards', 'the', 'brown', 'cow']:
            self.vocabulary.tokenize(word)
            
    def test_tokens(self):
        self.assertEqual('cow',self.vocabulary.get_word(10))
        
    def test_count(self):                       
        self.assertEqual(3,self.vocabulary.get_count(0))   # Number of 'the's
        self.assertEqual(2,self.vocabulary.get_count(2))   # Number of 'brown's
        self.assertEqual(1,self.vocabulary.get_count(3))   # Number of 'fox's  
        
class TestVocabularyWithSOS_EOS(TestCase):
    '''
    Test case with sentence tokens
    '''    
    def test_SOS_EOS(self):
        vocabulary = Vocabulary(sentence_tokens=True)
        self.assertCountEqual('<SOS>',vocabulary[0])
        self.assertCountEqual('<EOS>',vocabulary[1])
        for word in ['the', 'quick', 'brown','fox', 'jumps', 'over', 'the', 'lazy', 'dog',
                          'that', 'guards', 'the', 'brown', 'cow']:
            vocabulary.tokenize(word)
        self.assertEqual('cow',vocabulary.get_word(12))   
        vocabulary.save('foo.pkl')
        v2 = Vocabulary.create('foo.pkl')
        self.assertEqual(13,len(v2))
        
if __name__ == '__main__':
    main()
