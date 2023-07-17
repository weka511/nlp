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
from csv import writer
from pathlib import Path
from time import time
import numpy as np
from sklearn.cluster import KMeans
from skipgram import Word2Vec, Vocabulary,Index2Word
from word2vec2 import create_file_name

def create_arguments():
    '''
    Parse command line arguments

    Returns:
       'args' object
    '''

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--load', default='word2vec2', help='File name to load weights')
    parser.add_argument('--data', default='./data', help='Path to data files')
    parser.add_argument('--vocabulary', default='vocabulary', help='File name for vocabulary')
    parser.add_argument('--k', type=int, default=8, help='Number of clusters')
    parser.add_argument('--clusters', default = Path(__file__).stem, help='File name for clustering results')
    parser.add_argument('--vectors', choices=['words','contexts','both'], help='Establish whether we are clustering on words, context, or both')
    return parser.parse_args()

def get_vectors(model,vector_type='both'):
    '''
    Establish whether we are clustering on words, context, or both

    Parametere:
        model
        vector_type
    '''
    match vector_type:
        case 'words':
            return model.w
        case 'contexts':
            return model.c
        case _:
            return model.w + model.c

if __name__=='__main__':
    start  = time()
    args = create_arguments()
    vocabulary = Vocabulary()
    vocabulary_file = create_file_name(args.vocabulary,path=args.data)
    vocabulary.load(vocabulary_file)
    words = Index2Word(vocabulary)
    print (f'Loaded {vocabulary_file}--{len(vocabulary)} words')
    model = Word2Vec()
    model_name = create_file_name(args.load,path=args.data)
    model.load(model_name)
    print (f'Loaded {model_name}')
    vectors = get_vectors(model,args.vectors)
    kmeans = KMeans(n_clusters=args.k, random_state=0, n_init='auto').fit(vectors)
    Distances = kmeans.transform(vectors)
    clusters_file = create_file_name(args.clusters,ext='csv',path=args.data)
    with open(clusters_file,'w', newline='') as out:
        clusters = writer(out)
        for i in range(args.k):
            print (f'Cluster {i}')
            words_and_distances = []
            for j in range(len(kmeans.labels_)):
                if kmeans.labels_[j]==i:
                    words_and_distances.append((words.get_word(j),Distances[j,i]))
            words_and_distances.sort(key=lambda tup: tup[1])
            for word,distance in words_and_distances:
                try:
                    clusters.writerow([i,word,distance])
                except UnicodeEncodeError as encode_error:
                    print (encode_error)

    print (f'Wrote {clusters_file}')
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
