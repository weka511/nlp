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

'''Cluster word2vec data'''

from argparse import ArgumentParser
from time import time
import numpy as np
from sklearn.cluster import KMeans
from skipgram import Word2Vec, Vocabulary,Index2Word
from word2vec2 import create_file_name

if __name__=='__main__':
    start  = time()
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--load', default='word2vec2', help='File name to load weights')
    parser.add_argument('--data', default='.', help='Path to data files')
    parser.add_argument('--vocabulary', default='vocabulary', help='File name for vocabulary')
    parser.add_argument('--n', type=int, default=8, help='Number of clusters')
    args = parser.parse_args()
    vocabulary = Vocabulary()
    vocabulary_file = create_file_name(args.vocabulary,path=args.data)
    vocabulary.load(vocabulary_file)
    words = Index2Word(vocabulary)
    print (f'Loaded {vocabulary_file}--{len(vocabulary)} words')
    model = Word2Vec()
    model_name = create_file_name(args.load,path=args.data)
    model.load(model_name)
    print (f'Loaded {model_name}')
    kmeans = KMeans(n_clusters=args.n, random_state=0, n_init='auto').fit(model.w)
    for i in range(args.n):
        print (f'Cluster {i}')
        for j in range(len(kmeans.labels_)):
            if kmeans.labels_[j]==i:
                print (words.get_word(j))

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
