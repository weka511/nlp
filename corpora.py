#!/usr/bin/env python

#   Copyright (C) 2023 Simon Crase

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

'''Template for python script using pytorch'''

# https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/

from argparse import ArgumentParser
from os import remove
from os.path import exists, join
from pathlib import Path
from time import time
from xml.dom import minidom
from xml.parsers.expat import ExpatError
import zipfile as zf
from matplotlib.pyplot import figure,show
import numpy as np

def generate_from_zipped_xml(dataset):
    with zf.ZipFile(dataset) as zipfile:
        for file_name in zipfile.namelist():
            print (file_name)
            if file_name.endswith('/'): continue
            path = zf.Path(zipfile, at=file_name)
            try:
                contents = path.read_text(encoding='UTF-8')
                doc = minidom.parseString(contents)
                post = doc.getElementsByTagName('post')
                # print (contents)
                for p in post:
                    print (p.firstChild.nodeValue)
            except UnicodeDecodeError as err:
                print (err)
            except ExpatError as err:
                print (err)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='./data', help='Path to data files (default: %(default)s)')
    parser.add_argument('--figs', default='./figs', help='Path to plot files (default: %(default)s)')
    parser.add_argument('--plot', default = Path(__file__).stem)
    parser.add_argument('--save', default = Path(__file__).stem)
    parser.add_argument('--dataset', default='...', help='Name of dataset (default: %(default)s)')
    parser.add_argument('--show', default=False, action='store_true')
    parser.add_argument('--N', default=100,type=int, help='Number of iterations (default: %(default)s)')
    parser.add_argument('--n', default=10,type=int)
    parser.add_argument('--freq', default=5,type=int, help='Controls how frequently progress will be shown (default: %(default)s)')
    return parser.parse_args()

if __name__=='__main__':
    start  = time()
    args = parse_args()
    generate_from_zipped_xml(join(args.data,args.dataset))
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
