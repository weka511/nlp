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
Construct an index for all file in example set, 
so we can use a ramdomizing data loader
'''

from argparse import ArgumentParser
from csv import reader,writer,QUOTE_MINIMAL
from pathlib import Path
from time import time

__version__ = '1.0'
__author__ = 'Simon Crase'

def parse_args():
    '''
    Parse command line arguments
    '''
    data = './data'
    examples = 'examples'    
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=data, help=f'Path to data files [{data}]')
    parser.add_argument('--examples', default=examples, help=f'Path to examples files [{examples}]')
    parser.add_argument('--verbose',default=False,action='store_true',help='Used to print counts')
    return parser.parse_args()

def get_line_count(path):
    '''
    Count lines in a csv file
    '''
    n = 0
    with open(path, newline='') as in_file:
        for row in reader(in_file, delimiter=','):
            n  += 1
    return n
            
def main():
    '''
    Read all files in example set and create index file
    '''
    start  = time()
    args = parse_args()
    root_dir = Path(args.data) / args.examples
    out_file_path = (root_dir / args.examples).with_suffix('.csv')
    with open(out_file_path, 'w', newline='') as out_file:
        out = writer(out_file, delimiter=',', quotechar='|', quoting=QUOTE_MINIMAL)
        for path in root_dir.rglob("*.csv"):
            if not path.is_file(): continue
            if out_file_path == path: continue    # We don't want to include index file in our counts
            out.writerow([path,get_line_count(path)])
            if args.verbose:
                print (path,get_line_count(path))
            
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    
if __name__=='__main__':
    main()
