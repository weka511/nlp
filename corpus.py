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
    Read from a corpus
'''

from argparse import ArgumentParser
from glob import glob
from os.path import join
from pathlib import Path
from time import time

import numpy as np
import nltk
from nltk.corpus.reader.bnc import BNCCorpusReader

from tokenizer import generate_sentences, generate_text, generate_tokens, Token

class BNC:
    '''
        Read from BNC corpus
    '''    
    def __init__(self,root = r'C:\Users\weka5\nlp\data\2554\download\Texts'):
        nltk.data.path.append(r'C:\Users\weka5\nlp\data\2554\download')
        self.bnc = BNCCorpusReader(root=root,fileids= r'[A-K]/\w*/\w*\.xml')        
    
    def filenames(self):
        for filename in self.bnc.fileids():
            yield filename
            
    def words(self):
        for word in self.bnc.words():
            yield word
            
    def sentences(self,filename='A/A0/A00.xml'):
        for sentence in self.bnc.sents(filename):
            yield sentence

def create_sentence_generator(file_pattern,data):
    match file_pattern[0]:
        case '*BNC*':
            bnc = BNC()
            return bnc.sentences()
        case _:
            return generate_sentences(
                generate_tokens(
                    generate_text(
                        file_names=[globbed for name in file_pattern for globbed in glob(join(data, name))]
                    )))   

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-n','--n',type=int,default=12)
    return parser.parse_args()
    
def main():
    start  = time()
    args = parse_args()
 
    bnc = BNC()
    for i,file in enumerate(bnc.filenames()):
        print (file)
        if i > args.n: break
  
    for i,word in enumerate(bnc.words()):
        print (word)
        if i > args.n: break
 
    for i,sentence in enumerate(bnc.sentences()):
        print (sentence)
        if i > args.n: break
    
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    
if __name__=='__main__':
    main()
