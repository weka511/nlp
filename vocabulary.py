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

from unittest import main, TestCase, skip

class Vocabulary:
    '''
    This class maps between words and tokens
    '''
    def __init__(self):
        self.symbols = []
        self.token = {}
        self.counts = []
        
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

 
    
class TestVocabulary(TestCase):
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
        
if __name__ == '__main__':
    main()
