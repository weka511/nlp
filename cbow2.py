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
    Read sentences from a corpus
'''

from argparse import ArgumentParser
import csv
from os.path import join
from pathlib import Path
from time import time

import numpy as np
import nltk
from nltk.corpus.reader.bnc import BNCCorpusReader

from vocabulary import Vocabulary
from shared.utils import Logger, user_has_requested_stop, get_seed
from tokenizer import generate_sentences, generate_text, generate_tokens, Token

class BNC:
    '''
    Read text from BNC corpus
    '''    
    def __init__(self,root = r'./data/2554/download\Texts'):
        nltk.data.path.append(r'./data\2554\download')
        self.bnc = BNCCorpusReader(root=root,fileids= r'[A-K]/\w*/\w*\.xml')        
    
    def filenames(self):
        '''
        This generator is used to iterate through filenames
        '''
        for filename in self.bnc.fileids():
            yield filename
            
    def words(self):
        for word in self.bnc.words():
            yield word
            
    def sentences(self,path):
        '''
        This generator is used to iterate through sentences in a specified file
        
        Parameters:
            filename
        '''
        for sentence in self.bnc.sents(fileids=path):
            yield sentence

class ExampleSet:
    def __init__(self,window_size=2):
        self.vocabulary = Vocabulary()
        self.SOS = self.vocabulary.tokenize('<SOS>')
        self.EOS = self.vocabulary.tokenize('<EOS>')
        self.window_size = window_size
        
    def build(self,sentence,out_file):      
        self.__accumulate__((
                            [self.SOS] +
                            [self.vocabulary.tokenize(word) for word in sentence] +
                            [self.EOS]   
                            ),out_file)


        
    def __accumulate__(self, tokens: [int],out_file):
        '''
        Convert a sequence of tokens, representing one sentence, to 
        words and contexts
        
        Parameters:
            tokens
        '''
        writer = csv.writer(out_file, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)          
        start = 0
        end = start + 2 * self.window_size + 1
        n_entries = len(tokens) - end + 1
        while end <= len(tokens):
            run = [tokens[i] for i in range(start, end)]
            words = run[self.window_size]
            context = run[:self.window_size] + run[self.window_size + 1:]
            writer.writerow([words]+context)
            start += 1
            end += 1
    

def parse_args():
    data = './data'
    logs = './logs'
    window_size = 4    
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=data, help=f'Path to data files [{data}]')
    parser.add_argument('--logs', default=logs, help=f'Location for storing log files [{logs}]')
    parser.add_argument('-o', '--output', default=None, required=True, help='File name for storing results')
    parser.add_argument('-m', '--window_size', type=int, default=window_size, 
                                help=f'Half size of window (context extends left and right) [{window_size}]')    
    return parser.parse_args()
    
def main():
    start  = time()
    args = parse_args()
    with Logger(Path(__file__).stem, path=args.logs) as _:
        out = Path(f'{args.data}/{args.output}')
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Writing reults to {out}')
        bnc = BNC()
        for path in bnc.filenames():
            Logger.get_instance().log(f'{__file__} {Logger.get_line()} Opened {path}')
            out_file_path = (out / path).with_suffix('.csv')
            out_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file_path,'w',newline='') as out_file:
                examples = ExampleSet(window_size=args.window_size)
                for sentence in bnc.sentences(path=path):
                    examples.build(sentence,out_file)
      
        elapsed = time() - start
        minutes = int(elapsed/60)
        seconds = elapsed - 60*minutes
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Elapsed Time {minutes} m {seconds:.2f} s')
    
if __name__=='__main__':
    main()
