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
from os.path import join
from time import time
from matplotlib.pyplot import figure, show
import numpy as np
from numpy.random import default_rng
from skipgram import Vocabulary, ExampleBuilder, Tower, Optimizer, Word2Vec, LossCalculator, Index2Word
from tokenizer import read_text, extract_sentences, extract_tokens

def read_training_data(file_name):
    '''
    Read file containing examples for training

    Parameters:
        file_name    Name of file that is to be read

    Returns:
       A numpy array, each word consisting of a word, context, and an indicator of +/-
    '''
    def count_rows():
        '''
        Establish number of rows required for array
        '''
        with open(file_name, newline='') as f:
            return len(f.readlines())

    training_data = np.empty((count_rows(),3),dtype=np.int64)
    with open(file_name, newline='') as csvfile:
        examples = reader(csvfile)
        for i,row in enumerate(examples):
            training_data[i,:] = np.array([int(s) for s in row],dtype=np.int64)
    return training_data

def create_vocabulary(docnames):
    '''
    Build vocabulary first, so we have frequencies

    Parameters:
        docnames  List of all documents to be read

    Returns:
        Vocabulary built from all documents

    '''
    Product = Vocabulary()

    for sentence in extract_sentences(extract_tokens(read_text(file_names = docnames))):
        Product.parse(sentence)

    return Product

def create_file_name(name,ext='npz',path=None):
    '''
    Ensure that file name has appropriate extension

    Parameters:
        name      Basic file name
        ext       Extension to be used if 'name' lacks extension
        path      Path to file
    Returns:
        fully qualified file name

    '''
    file_name = name if name.endswith(ext) else f'{name}.{ext}'
    return file_name if path==None else join(path,file_name)

def create_arguments():
    '''
    Parse command line arguments

    Returns:
       'args' object
    '''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['create', 'train', 'test'],
                        help='''
                        Action to be performed: create training examples from corpus;
                        train weights using examples; test weights
                        ''')
    parser.add_argument('--seed', type=int,default=None, help='Used to initialize random number generator')
    parser.add_argument('--show', default=False, action='store_true', help='display plots')
    parser.add_argument('--examples', default='examples.csv', help='File name for training examples')
    parser.add_argument('--vocabulary', default='vocabulary', help='File name for vocabulary')
    parser.add_argument('--data', default='./data', help='Path to data files')

    group_create = parser.add_argument_group('create', 'Parameters for create')
    group_create.add_argument('docnames', nargs='*', help='A list of documents to be processed')
    group_create.add_argument('--width', '-w', type=int, default=2, help='Window size for building examples')
    group_create.add_argument('--k', '-k', type=int, default=5, help='Number of negative examples for each positive')

    group_train = parser.add_argument_group('train', 'Parameters for train')
    group_train.add_argument('--minibatch', '-m', type=int, default=64, help='Minibatch size')
    group_train.add_argument('--dimension', '-d', type=int, default=64, help='Dimension of word vectors')
    group_train.add_argument('--N', '-N', type=int, default=2048, help='Number of iterations')
    group_train.add_argument('--eta', '-e', type=float, default=0.05, help='Starting value for learning rate')
    group_train.add_argument('--ratio', '-r', type=float, default=0.01, help='Final learning rate as a fraction of the first')
    group_train.add_argument('--tau', '-t', type=int, default=512, help='Number of steps to decrease learning rate')
    group_train.add_argument('--plot', default='word2vec2', help='Plot file name')
    group_train.add_argument('--save', default='word2vec2', help='File name to save weights')
    group_train.add_argument('--resume', default=False, action='store_true', help='Resume training')
    group_train.add_argument('--checkpoint', default='checkpoint', help='File name to save weights at checkpoint')
    group_train.add_argument('--freq', type=int, default=25, help='Save checkoint every FREQ iteration')

    group_test = parser.add_argument_group('test', 'Parameters for test')
    group_test.add_argument('--load', default='word2vec2', help='File name to load weights')
    group_train.add_argument('--L', '-L', type=int, default=6, help='Number of words to compare')

    return parser.parse_args()

if __name__=='__main__':
    start  = time()
    args = create_arguments()
    rng = default_rng(args.seed)
    match args.action:
        case 'create':
            docnames = [doc for pattern in args.docnames for doc in glob(join(args.data,pattern))]
            vocabulary = create_vocabulary(docnames)
            word2vec = ExampleBuilder(k=args.k, width=args.width)
            tower = Tower(ExampleBuilder.normalize(vocabulary),rng=rng)
            examples_file = create_file_name(args.examples,ext='csv',path=args.data)
            with open(examples_file,'w', newline='') as out:
                examples = writer(out)
                n = 1
                for sentence in extract_sentences(extract_tokens(read_text(file_names = docnames))):
                    indices = vocabulary.parse(sentence)
                    for word,context,y in word2vec.generate_examples([indices],tower):
                        examples.writerow([word,context,y])
                        n += 1
            print (f'Saved {n} examples to {examples_file}')
            vocabulary_file = create_file_name(args.vocabulary,path=args.data)
            vocabulary.save(vocabulary_file)
            print (f'Saved vocabulary of {len(vocabulary)} words to {vocabulary_file}')

        case 'train':
            data = read_training_data(join(args.data,args.examples))
            model = Word2Vec()
            if args.resume:
                model.load(create_file_name(args.load,path=args.data))
            else:
                model.build(data[:,0].max()+1,n=args.dimension,rng=rng)
            loss_calculator = LossCalculator(model,data)
            optimizer = Optimizer.create(model,data,loss_calculator,
                                         m = args.minibatch,N = args.N,eta0 = args.eta,
                                         final_ratio=args.ratio, tau = args.tau, rng=rng)
            optimizer.optimize()
            save_file_name = create_file_name(args.save,path=args.data)
            model.save(save_file_name)
            print (f'Saved weights in {save_file_name}')
            fig = figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(range(len(optimizer.log)),optimizer.log)
            ax.ticklabel_format(style='plain',axis='x',useOffset=False)
            ax.set_title(f'Minibatch={args.minibatch}, dimension={args.dimension}')
            ax.set_xlabel('Step number')
            ax.set_ylabel('Loss')
            fig.savefig(args.plot)

        case test:
            model = Word2Vec()
            model_name = create_file_name(args.load,path=args.data)
            model.load(model_name)
            print (f'Loaded {model_name}')
            vocabulary = Vocabulary()
            vocabulary_file = create_file_name(args.vocabulary,path=args.data)
            vocabulary.load(vocabulary_file)
            words = Index2Word(vocabulary)
            print (f'Loaded {vocabulary_file}')
            NormalizedInnerProductsW = np.abs(model.create_productsW())
            InnerProductsWC = np.abs(model.create_productsWC())
            m,_ = NormalizedInnerProductsW.shape
            fig = figure()
            ax1 = fig.add_subplot(2,1,1)
            n,bins,_ = ax1.hist([NormalizedInnerProductsW[i,j] for i in range(m) for j in range(i)],bins=20)
            ax1.set_title(f'Weights')
            ax2 = fig.add_subplot(2,1,2)
            n,bins,_ = ax2.hist([InnerProductsWC[i,j] for i in range(m) for j in range(i)],bins=20)
            ax2.set_title(f'Weights by Context')
            for i in range(m):
                print ( f'{words.get_word(i)}')
                nearest_neighbours = [j for j in np.argpartition(InnerProductsWC[i,:], -args.L)[-args.L:] if i!=j]
                for j in nearest_neighbours:
                    print ( f'\t{words.get_word(j)} ({InnerProductsWC[i,j]:.4f})')


            fig.savefig(args.plot)

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
