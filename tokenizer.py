#!/usr/bin/env python

#    Copyright (C) 2021-2026 Simon A. Crase   simon@greenweaves.nz
#
#    This is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This software is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>

'''A library for extract tokens from a text'''

from re import split
from unittest import TestCase,main

class Token:
    Apostrophe = "'"
    Period = '.'
    Apostrophe2 = '’'  #FIXME - Issue #45
    
def generate_text(file_names : [str] = []):
    '''
    Generator for reading text from a corpus. It allows us to read the file, one line at a time.

    Parameters:
        file_names   One or more text files that make up corpus
    '''
    if len(file_names) == 0: raise Exception('generate_text() needs a list of file names')
    
    for file_name in file_names:
        with open(file_name, encoding='utf-8') as text_file:
            for line in text_file:
                yield line.strip()

def consolidate_apostrophes(tokens : [str]):
    '''
    A generator to consolidate apostrophes. It handle words such as "we've"

        [..."we", "'", "ve"...] -> [..."we've"...]
        
    Parameters:
        tokens    A list of words and punctuation symbols
    '''
    if len(tokens) == 0: return
    
    word = tokens[0]
    i = 1
    while i < len(tokens):
        if tokens[i] == Token.Apostrophe or tokens[i] == Token.Apostrophe2:
            i += 1     # point beyond apostrophe
            if i < len(tokens):
                yield f'{word}{Token.Apostrophe}{tokens[i]}'
                i += 1   # point beyond part following the apostrophe
                try:
                    word = tokens[i]
                except IndexError:     # Handle case where apostrophized word is the last token
                    return
                i += 1
            else:
                yield  f'{word}{Token.Apostrophe}'
        else:
            yield word
            word = tokens[i]
            i += 1
            
    yield word 
    
def generate_tokens(text: [str]):
    '''
    Extract tokens from text
    
    Parameters:
        text     A list of strings of text
    '''
    for line in text:
        Tokens = [token.strip() for token in split(r'(\W+)', line.strip()) if len(token.replace(' ', '')) != 0]
        for token in consolidate_apostrophes(Tokens):
            if token == 's':
                print (line)
            yield token.lower()


def generate_sentences(tokens : [str]):
    '''
    Split list of tokens into list of lists
    
    Parameters:
        tokens   This list that will be split
    '''
    sentence = []
    for token in tokens:
        if token == Token.Period:
            yield sentence
            sentence = []
        else:
            sentence.append(token)

class TestApostropheTest(TestCase):
    def test1(self):
        '''
        Test for Issue #45
        '''
        tokens = list(generate_tokens([
            "Buccleuch, but the actual founder of my line was my grandfather's"
        ]))
        self.assertEqual(12,len(tokens))
        self.assertEqual("grandfather's",tokens[-1])
        self.assertEqual("buccleuch",tokens[0])
     
    def test2(self):
        '''
        Test for Issue #45
        '''        
        tokens = list(generate_tokens([
            "The valley of ashes is bounded on one side by a small foul river, and,"
            " when the drawbridge is up to let barges through, the passengers"
            " on waiting trains can stare at the dismal scene for as long as half an hour."
            " There is always a halt there of at least a minute, and it was because"
            " of this that I first met Tom Buchanan’s mistress."
        ]))
        self.assertEqual("buchanan's",tokens[-3])
        
if __name__ == '__main__':
    main()
