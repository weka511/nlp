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

from argparse import ArgumentParser
from glob import glob
from time import time
import numpy as np
from tfidf import TfIdf, create_inner_products

if __name__=='__main__':
    start  = time()
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('docnames', nargs='+', help='A list of documents to be processed')
    parser.add_argument('--log', action='store_true', default=False)
    args = parser.parse_args()
    docnames = [doc for pattern in args.docnames for doc in glob(pattern)]
    print (docnames)
    words,tf_idf = TfIdf(docnames=docnames)
    tf_idf = tf_idf/np.linalg.norm(tf_idf,axis=0,keepdims=True)
    if args.log:
        for i,word in enumerate(words):
            if np.any(tf_idf[i,:]>0):
                print (word, tf_idf[i,:])
    D =  create_inner_products(tf_idf)
    print(D)
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
