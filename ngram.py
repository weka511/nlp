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

'''Exercises from Chapter 3 of Jurafsky and Martin'''

from argparse import ArgumentParser
from glob import glob
from os.path import join
from time import time
import numpy as np
from tokenizer import generate_sentences, generate_text, generate_tokens

class Ngram:
    def __init__(self,n):
        self.n = n
    def build(self,sentence):
        pass
    
def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default=None, nargs='+', help='Name(s) of corpus file(s)')
    parser.add_argument('--data', default='./data')
    parser.add_argument('-n', '--n',default=3, type=int)
    return parser.parse_args()
    
def main():
    start  = time()
    args = parse_args()
    ngram = Ngram(args.n)
    file_names = [globbed for name in args.corpus for globbed in glob(join(args.data, name))]
    for sentence in generate_sentences(generate_tokens(generate_text(file_names=file_names))):
        ngram.build(sentence)
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    
if __name__=='__main__':
    main()
