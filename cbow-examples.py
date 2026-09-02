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
    Read sentences from a corpus and generate words 
    and contexts to we can train CBOW
'''

from argparse import ArgumentParser
from os.path import join
from pathlib import Path
from time import time
import numpy as np
from vocabulary import Vocabulary
from shared.utils import Logger, user_has_requested_stop, get_seed
from cbow2 import ExampleSet, BNC

__version__ = '1.1'
__author__ = 'Simon Crase'

def parse_args():
    data = './data'
    logs = './logs'
    window_size = 4
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=data, help=f'Path to data files [{data}]')
    parser.add_argument('--logs', default=logs, help=f'Location for storing log files [{logs}]')
    parser.add_argument('--stem', default=False,action='store_true',help='Set this to use stems instead of full words')
    parser.add_argument('-o', '--output', default=None, required=True, help='File name for storing results')
    parser.add_argument('-m', '--window_size', type=int, default=window_size,
                        help=f'Half size of window (context extends left and right) [{window_size}]')
    parser.add_argument('--component',
                        default=r'/\w*\.xml',
                        choices=['aca','dem','fic','news'],
                        help='database component')
    return parser.parse_args()

def main():
    '''
    Read sentences from a corpus and generate words 
    and contexts to we can train CBOW
    '''    
    start = time()
    args = parse_args()
    total_examples = 0
    with Logger(Path(__file__).stem, path=args.logs) as _:
        out_root_path = Path(f'{args.data}/{args.output}')
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Writing results to {out_root_path}')
        bnc = BNC(component=args.component)
        vocabulary = Vocabulary()
        for path in bnc.filenames():
            Logger.get_instance().log(f'{__file__} {Logger.get_line()} Opened {path}')
            out_file_path = (out_root_path / path).with_suffix('.csv')
            out_file_path.parent.mkdir(parents=True, exist_ok=True)
            examples = ExampleSet(window_size=args.window_size, vocabulary=vocabulary)
            with open(out_file_path, 'w', newline='') as out_file:
                for sentence in bnc.sentences(path=path,stem=args.stem):
                    examples.build(sentence, out_file)
            Logger.get_instance().log(f'{__file__} {Logger.get_line()} Vocabulary contains {len(vocabulary)} words')
            Logger.get_instance().log(f'{__file__} {Logger.get_line()} There are {len(examples)} examples')
            total_examples += len(examples)
            
            if user_has_requested_stop():
                break
             
        vocabulary_path = (out_root_path / 'vocabulary').with_suffix('.pkl')           
        vocabulary.save(vocabulary_path)
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} There are {total_examples} examples')
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Saved vocabulary in {vocabulary_path}')
        
        elapsed = time() - start
        minutes = int(elapsed / 60)
        seconds = elapsed - 60 * minutes
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Elapsed Time {minutes} m {seconds:.2f} s')

if __name__ == '__main__':
    main()
