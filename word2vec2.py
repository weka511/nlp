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


'''Skipgrams as described in Chapter 6 of Jurafsky & Martin'''

from argparse import ArgumentParser
from csv import reader, writer
from glob import glob
from os import system
from time import time
import numpy as np
from numpy.random import default_rng
from skipgram import Vocabulary, ExampleBuilder, Tower, Optimizer, Word2Vec
from tokenizer import read_text, extract_sentences, extract_tokens

def read_training_data(file_name):
    def count_entries():
        count = 0
        with open(file_name, newline='') as csvfile:
            examples = reader(csvfile)
            for row in examples:
                count += 1
        return count

    training_data = np.empty((count_entries(),3),dtype=np.int64)
    with open(file_name, newline='') as csvfile:
        examples = reader(csvfile)
        for i,row in enumerate(examples):
            training_data[i,:] = np.array([int(s) for s in row],dtype=np.int64)
    return training_data

def create_vocabulary(docnames):
    '''
    Build vocabulary first, so we have frequencies
    '''
    Product = Vocabulary()

    for sentence in extract_sentences(extract_tokens(read_text(file_names = docnames))):
        Product.parse(sentence)

    return Product

if __name__=='__main__':
    start  = time()
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['create', 'train'])
    parser.add_argument('docnames', nargs='*', help='A list of documents to be processed')
    parser.add_argument('--examples', default='examples.csv', help='File name for training examples')
    parser.add_argument('--width', '-w', type=int, default=2, help='Window size for building examples')
    parser.add_argument('--k', '-k', type=int, default=2, help='Number of negative examples for each positive')
    parser.add_argument('--seed', type=int,default=None)
    args = parser.parse_args()
    rng = default_rng(args.seed)
    match args.action:
        case 'create':
            docnames = [doc for pattern in args.docnames for doc in glob(pattern)]
            vocabulary = create_vocabulary(docnames)
            word2vec = ExampleBuilder(k=args.k, width=args.width)
            tower = Tower(ExampleBuilder.normalize(vocabulary),rng=rng)
            with open(args.examples,'w', newline='') as out:
                examples = writer(out)
                for sentence in extract_sentences(extract_tokens(read_text(file_names = docnames))):
                    indices = vocabulary.parse(sentence)
                    for word,context,y in word2vec.generate_examples([indices],tower):
                        examples.writerow([word,context,y])

        case 'train':
            data = read_training_data(args.examples)
            model = Word2Vec()
            model.build(data[:,0].max(),rng=rng)
            optimizer = Optimizer.create(model,data,rng=rng)
            optimizer.optimize()


    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
