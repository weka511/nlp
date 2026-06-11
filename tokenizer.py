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

class Token:
    Apostrophe = "'"
    Period = '.'
    
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
        tokens    A list of words and punctutaion symbols
    '''
    if len(tokens) == 0: return
    
    word = tokens[0]
    i = 1
    while i < len(tokens):
        if tokens[i] == Token.Apostrophe:
            i += 1     # point beyond apostrophe
            if i < len(tokens) - 1:
                yield f'{word}{Token.Apostrophe}{tokens[i]}'
                i += 1   # point beyond part following the apostrophe
                word = tokens[i]
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
            yield token.lower()


def generate_sentences(Tokens : [str]):
    '''
    Split list of tokens into list of lists
    
    Parameters:
        Tokens
    '''
    sentence = []
    for token in Tokens:
        if token == Token.Period:
            yield sentence
            sentence = []
        else:
            sentence.append(token)

def main():
    for sentence in generate_sentences(generate_tokens(generate_text(file_names=['data/gatsby1.txt']))):
        print(sentence)

if __name__ == '__main__':
    main()
