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
from csv import writer
from glob import glob
from time import time
import numpy as np
from skipgram import Vocabulary, ExampleBuilder, Tower
from tokenizer import read_text, extract_sentences, extract_tokens

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

            with open(args.examples,'w', newline='') as out:
                examples = writer(out)
                for sentence in extract_sentences(extract_tokens(read_text(file_names = docnames))):
                    indices = vocabulary.parse(sentence)
                    for word,context in word2vec.generate_positive_examples([indices]):
                        examples.writerow([word,context,+1])
                for word,context in  word2vec.create_negative_examples(ExampleBuilder.normalize(vocabulary)):
                    examples.writerow([word,context,-1])

        case 'train':
            print ('Not implemented')


    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
