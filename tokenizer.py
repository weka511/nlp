#!/usr/bin/env python

#    Copyright (C) 2021-2023 Simon A. Crase   simon@greenweaves.nz
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

'''A library for tokenizing text'''

from re import split


def read_text(file_names = []):
    '''
    read_text

    Generator for reading text from a corpus. It allows us to read the file, one line at a time.

    Parameters:
        file_names   One or more text files that make up corpus
    '''
    if len(file_names)==0:
        raise Exception('read_text() needs a list of file names')
    for file_name in file_names:
        with open(file_name,encoding='utf-8') as text_file:
            for line in text_file:
                yield line.strip()


def extract_tokens(text):
    '''
    extract_tokens

    Extract tokens from text
    '''

    def consolidate_apostrophes(tokens):
        '''
        consolidate_apostrophes

        Handle wordds such as "we,ve"

            [..."we", "'", "ve"...] -> [..."we've"...]
        '''
        Result = []
        word = tokens[0]
        i = 1
        while i<len(tokens):
            if tokens[i]=="'":
                i += 1
                if i<len(tokens)-1:
                    word = f"{word}'{tokens[i]}"
                    Result.append(word)
                    i += 1
                    word = tokens[i]
                else:
                    word = f"{word}'"
                    Result.append(word)
            else:
                Result.append(word)
                word = tokens[i]
                i+=1
        Result.append(word)
        return Result

    for line in text:
        Tokens = [token.strip() for token in split(r'(\W+)',line.strip()) if len(token.replace(' ','')) != 0]
        if len(Tokens)==0: continue
        for token in consolidate_apostrophes(Tokens):
            yield token.lower()


def extract_sentences(Tokens):
    '''
    extract_sentences

    split list of tokens into list of lists
    '''
    sentence = []
    for token in Tokens:
        if token == '.':
            yield sentence
            sentence = []
        else:
            sentence.append(token)

if __name__=='__main__':
    for sentence in extract_sentences(extract_tokens(read_text(file_names = ['chapter1.txt']))):
        print (sentence)
