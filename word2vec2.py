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
from os import remove, system
from os.path import exists, join
from sys import exit
from time import time
from matplotlib.pyplot import figure, show, rcParams
import numpy as np
from numpy.random import default_rng
from skipgram import Vocabulary, ExampleBuilder, Tower, Optimizer, Word2Vec, LossCalculator, Index2Word
from tokenizer import read_text, extract_sentences, extract_tokens
from warnings import warn

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

    with open(file_name, newline='') as csvfile:
        paths = []
        examples = reader(csvfile)
        i = 0
        skip = 0
        in_examples = False
        for row in examples:
            if in_examples:
                training_data[i,:] = np.array([int(s) for s in row],dtype=np.int64)
                i += 1
            else:
                match row[0]:
                    case 'k':
                        k = int(row[1])
                        skip += 1
                    case 'width':
                        width = int(row[1])
                        skip += 1
                    case 'word':
                        skip += 1
                        in_examples = True
                        training_data = np.empty((count_rows()-skip,3),dtype=np.int64)
                    case _:
                        skip += 1
                        paths.append(row[0])

    return k,width,paths,training_data

def create_vocabulary(docnames,verbose=False):
    '''
    Build vocabulary first, so we have frequencies

    Parameters:
        docnames  List of all documents to be read

    Returns:
        Vocabulary built from all documents

    '''
    Product = Vocabulary()

    for sentence in extract_sentences(extract_tokens(read_text(file_names = docnames))):
        Product.parse(sentence,verbose=verbose)

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
    parser.add_argument('command', choices=['create', 'train', 'test'],
                        help='''
                        Command to be executed by program:
                            create training examples from corpus;
                            train weights using examples;
                            test weights
                        ''')
    parser.add_argument('--seed', type=int,default=None, help='Used to initialize random number generator')
    parser.add_argument('--show', default=False, action='store_true', help='display plots')
    parser.add_argument('--examples', default='examples.csv', help='File name for training examples')
    parser.add_argument('--vocabulary', default='vocabulary', help='File name for vocabulary')
    parser.add_argument('--data', default='./data', help='Path to data files')
    parser.add_argument('--figs', default='./figs', help='Path to save plots')

    group_create = parser.add_argument_group('create', 'Parameters for create')
    group_create.add_argument('docnames', nargs='*', help='A list of documents to be processed')
    group_create.add_argument('--width', '-w', type=int, default=2, help='Window size for building examples')
    group_create.add_argument('--k', '-k', type=int, default=2, help='Number of negative examples for each positive')
    group_create.add_argument('--verbose', default=False, action='store_true')

    group_train = parser.add_argument_group('train', 'Parameters for train')
    group_train.add_argument('--minibatch', '-m', type=int, default=64, help='Minibatch size')
    group_train.add_argument('--dimension', '-d', type=int, default=64, help='Dimension of word vectors')
    group_train.add_argument('--N', '-N', type=int, default=2048, help='Number of iterations')
    group_train.add_argument('--eta', '-e', type=float, default=0.05, help='Starting value for learning rate')
    group_train.add_argument('--ratio', '-r', type=float, default=0.01, help='Final learning rate as a fraction of the first')
    group_train.add_argument('--tau', '-t', type=int, default=None, help='Number of steps to decrease learning rate')
    group_train.add_argument('--plot', default='word2vec2', help='Plot file name')
    group_train.add_argument('--save', default='word2vec2', help='File name to save weights')
    group_train.add_argument('--resume', default=False, action='store_true', help='Resume training')
    group_train.add_argument('--checkpoint', default='checkpoint', help='File name to save weights at checkpoint')
    group_train.add_argument('--freq', type=int, default=25, help='Report progress and save checkpoint every FREQ iteration')

    group_test = parser.add_argument_group('test', 'Parameters for test')
    group_test.add_argument('--load', default='word2vec2', help='File name to load weights')
    group_train.add_argument('--L', '-L', type=int, default=6, help='Number of words to compare')

    return parser.parse_args()

def establish_tau(tau,N=1000000,m = 128,M=1000000, target=0.03125):
    '''
    Fix to Issue #26: if tau has not been specified, fix it so most data points have been covered.

    Parameters:
        tau    From command line (may be None)
        N      Number of iterations
        m      Minibatch size
        M      Training dataset size
        target Target probability: we expect that no data has a higher probability than this of being covered
    '''
    if tau==None:
        ln_target = np.log(target)
        ratio = (M - m)/M
        ln_ratio = np.log(ratio)
        tau = int(ln_target/ln_ratio)
        if tau>N:
            warn(f'Calculated tau {tau} exceeds N {N}')

    return tau

def is_stopping(token='stop',message='Stopfile detected and deleted'):
    '''
    Used to allow a controlled stop using a stopfile.

    Parameters:
        token   Name of stopfile
        message Message to be shown when stopping

    Returns:
        True iff stopfile exists

    '''
    stopping = exists(token)
    if stopping:
        remove(token)
        print (message)
    return stopping

if __name__=='__main__':
    rcParams['text.usetex'] = True
    start  = time()
    args = create_arguments()
    rng = default_rng(args.seed)
    match args.command:
        case 'create':
            docnames = [doc for pattern in args.docnames for doc in glob(join(args.data,pattern))]
            vocabulary = create_vocabulary(docnames, verbose=args.verbose)
            word2vec = ExampleBuilder(k=args.k, width=args.width)
            tower = Tower(ExampleBuilder.normalize(vocabulary),rng=rng)
            examples_file = create_file_name(args.examples,ext='csv',path=args.data)
            with open(examples_file,'w', newline='') as out:
                examples = writer(out)
                for doc in docnames:
                    examples.writerow([doc])
                examples.writerow(['k',args.k])
                examples.writerow(['width',args.width])
                examples.writerow(['word','context','y'])

                for sentence in extract_sentences(extract_tokens(read_text(file_names = docnames))):
                    indices = vocabulary.parse(sentence)
                vocabulary_file = create_file_name(args.vocabulary,path=args.data)
                vocabulary.save(vocabulary_file)
                print (f'Saved vocabulary of {len(vocabulary)} words to {vocabulary_file}')

                n = 1
                for sentence in extract_sentences(extract_tokens(read_text(file_names = docnames))):
                    indices = vocabulary.parse(sentence)
                    for word,context,y in word2vec.generate_examples([indices],tower):
                        examples.writerow([word,context,y])
                        n += 1
            print (f'Saved {n} examples to {examples_file}')


        case 'train':
            if len(args.docnames)>0:   # Issue 21
                exit(f'Docnames {args.docnames} with train does not make sense')
            k,width,paths,data = read_training_data(join(args.data,args.examples))
            model = Word2Vec()
            if args.resume:
                model.load(create_file_name(args.load,path=args.data))
            else:
                model.build(data[:,0].max()+1,n=args.dimension,rng=rng)
            loss_calculator = LossCalculator(model,data)
            optimizer = Optimizer.create(model,data,loss_calculator,
                                         m=args.minibatch,N=args.N,eta=args.eta,final_ratio=args.ratio,
                                         tau=establish_tau(args.tau,N=args.N,m=args.minibatch,M=len(data)),rng=rng,
                                         checkpoint_file=create_file_name(args.checkpoint,path=args.data),freq=args.freq)
            eta,total_loss = optimizer.optimize(is_stopping=is_stopping)
            model.save(create_file_name(args.save,path=args.data), width=width,k=k,paths=paths,total_loss=total_loss,eta=eta)

            fig = figure()
            ax1 = fig.add_subplot(1,1,1)
            t = [args.freq*i for i in range(len(optimizer.log))]
            ax1.plot(t,optimizer.log,color='xkcd:red')
            ax1.ticklabel_format(style='plain',axis='x',useOffset=False)
            ax1.set_title(f'Minibatch={args.minibatch}, dimension={model.n}')
            ax1.set_xlabel('step')
            ax1.set_ylabel('Loss',color='xkcd:red')
            ax1.set_ylim(bottom=0)
            ax2 = ax1.twinx()
            ax2.plot(t,optimizer.etas,color='xkcd:blue')
            ax2.set_ylabel(r'$\eta$',color='xkcd:blue')
            ax2.set_ylim(bottom=0)
            fig.savefig(join(args.figs,args.plot))

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

            fig.savefig(join(args.figs,args.plot))

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
