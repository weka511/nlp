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
from skipgram import Vocabulary, ExampleBuilder, Tower
from tokenizer import read_text, extract_sentences, extract_tokens

def read_training_data(file_name):
    raw_data = []
    with open(file_name, newline='') as csvfile:
        examples = reader(csvfile)
        for row in examples:
            raw_data.append([int(s) for s in row])
    training_data = np.array(raw_data)
    index = np.full((training_data.max()+1,2),-1)
    m,n = training_data.shape
    for i in range(m):
        word_index = training_data[i,0]
        if index[word_index,0] == -1:
            index[word_index,0] = i
        index[word_index,1] = i

    return index,training_data


if __name__=='__main__':
    start  = time()
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('docnames', nargs='+', help='A list of documents to be processed')
    parser.add_argument('--action', choices=['create', 'train'], required=True)
    parser.add_argument('--examples', default='examples.csv')
    args = parser.parse_args()

    match args.action:
        case 'create':
            docnames = [doc for pattern in args.docnames for doc in glob(pattern)]
            vocabulary = Vocabulary()
            word2vec = ExampleBuilder()
            for sentence in extract_sentences(extract_tokens(read_text(file_names = docnames))):
                vocabulary.parse(sentence)
            tower = Tower(ExampleBuilder.normalize(vocabulary))
            with open(args.examples,'w', newline='') as out:
                examples = writer(out)

                for sentence in extract_sentences(extract_tokens(read_text(file_names = docnames))):
                    indices = vocabulary.parse(sentence)
                    for word,context,y in word2vec.generate_examples([indices],tower):
                        examples.writerow([word,context,y])
                    # for word,context in word2vec.generate_positive_examples([indices]):
                        # examples.writerow([word,context,+1])
                # for word,context in  word2vec.create_negative_examples(ExampleBuilder.normalize(vocabulary)):
                    # examples.writerow([word,context,-1])
            # system(f'sort {args.examples}  -g -o {args.examples}')

        case 'train':
            index,training_data = read_training_data(args.examples)
            print (index)


    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
