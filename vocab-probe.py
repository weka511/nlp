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

'''Investigate Issue #92: Vocabulary has too many tokens'''

from argparse import ArgumentParser
from pathlib import Path
from time import time
import numpy as np
from shared.utils import Logger, user_has_requested_stop, get_seed
from vocabulary import Vocabulary

def parse_args():
    parser = ArgumentParser(description=__doc__)
    data = './data'
    logs = './logs'
    examples = 'examples'
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=data, help=f'Path to data files [{data}]')
    parser.add_argument('--logs', default=logs, help=f'Location for storing log files [{logs}]')
    parser.add_argument('--examples', default=examples, help=f'Path to examples files [{examples}]')    
    return parser.parse_args()
    
def main():
    start  = time()
    args = parse_args()
    with Logger(Path(__file__).stem, path=args.logs) as _:
        root_dir = Path(args.data) / args.examples
        vocabulary = Vocabulary.create((root_dir / 'vocabulary').with_suffix('.pkl'))
        for symbol in sorted(vocabulary.symbols):
            try:
                Logger.get_instance().log(f'{__file__} {Logger.get_line()} {symbol}')
            except UnicodeEncodeError:
                pass
        elapsed = time() - start
        minutes = int(elapsed/60)
        seconds = elapsed - 60*minutes
        print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    
if __name__=='__main__':
    main()
