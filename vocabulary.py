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

class Vocabulary:
    '''
    Establish mapping between words and tokens
    '''
    def __init__(self):
        self.symbols = []
        self.token = {}
        
    def tokenize(self,word):
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
        
    def get_word(self,token):
        '''
        Look up the word that corresponds to a token
        
        Parameters:
            token      An index into symbol table
        '''
        return self.symbols[token]  
