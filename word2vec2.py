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


'''Train Skipgrams from supplied corpus. See Chapter 6 of Jurafsky & Martin'''

from argparse import ArgumentParser
from csv import reader, writer
from glob import glob
from os import remove, system
from os.path import exists, join
from pathlib import Path
from sys import exit, stderr
from time import time
from warnings import warn
from matplotlib.pyplot import figure, show, rcParams
import numpy as np
from numpy.random import default_rng
from skipgram import Vocabulary, ExampleBuilder, Tower, Optimizer, Word2Vec, LossCalculator, Index2Word
from corpora import Corpus

def read_training_data(file_name):
    '''
    Read file containing examples for training

    Parameters:
        file_name    Name of file that is to be read

    Returns:
       A numpy array, each row consisting of a word, context, and an indicator of +/-
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
    parser.add_argument('command', choices=['build', 'create', 'train', 'postprocess', 'extract', 'list'],
                        help='''
                        Command to be executed by program:
                            build vocabulary from corpus;
                            create training examples from corpus;
                            train weights using examples;
                            postprocess - compute matrix of distances between vectors;
                            extract - extract pairs of words, starting with closest
                        ''')
    parser.add_argument('--seed', type=int,default=None, help='Used to initialize random number generator')
    parser.add_argument('--show', default=False, action='store_true', help='display plots (default: %(default)s)')
    parser.add_argument('--examples', default='examples.csv', help='File name for training examples (default: %(default)s)')
    parser.add_argument('--vocabulary', default='vocabulary', help='File name for vocabulary (default: %(default)s)')
    parser.add_argument('--data', default='./data', help='Path to data files (default: %(default)s)')
    parser.add_argument('--figs', default='./figs', help='Path to save plots (default: %(default)s)')
    parser.add_argument('--logfile', default='logfile.txt', help='Path to save error messages (default: %(default)s)')

    group_build = parser.add_argument_group('build', 'Parameters for building vocabulary')
    group_build.add_argument('--n', '-n', type=int, default=None, help='Number of files from corpus')
    group_build.add_argument('--format', choices=['ZippedXml', 'Text'], default='ZippedXml', help='Format for corpus (default: ZippedXml)')

    group_create = parser.add_argument_group('create', 'Parameters for creating training examples')
    group_create.add_argument('docnames', nargs='*', help='A list of documents to be processed')
    group_create.add_argument('--width', '-w', type=int, default=2, help='Window size for building examples (default: %(default)s)')
    group_create.add_argument('--k', '-k', type=int, default=2, help='Number of negative examples for each positive (default: %(default)s)')
    group_create.add_argument('--verbose', default=False, action='store_true')

    group_train = parser.add_argument_group('train', 'Parameters for training weights')
    group_train.add_argument('--minibatch', '-m', type=int, default=64, help='Minibatch size (default: %(default)s)')
    group_train.add_argument('--dimension', '-d', type=int, default=64, help='Dimension of word vectors (default: %(default)s)')
    group_train.add_argument('--N', '-N', type=int, default=2048, help='Number of iterations (default: %(default)s)')
    group_train.add_argument('--eta', '-e', type=float, default=0.05,
                             help='Starting value for learning rate (default: %(default)s)')
    group_train.add_argument('--ratio', '-r', type=float, default=0.01,
                             help='Final learning rate as a fraction of the first (default: %(default)s)')
    group_train.add_argument('--tau', '-t', type=int, default=None,
                             help='Number of steps to decrease learning rate. if tau has not been specified,'
                             ' fix it so most data points have been covered')
    group_train.add_argument('--plot', default=Path(__file__).stem, help='Plot file name (default: %(default)s)')
    group_train.add_argument('--save', default=Path(__file__).stem, help='File name to save weights (default: %(default)s)')
    group_train.add_argument('--resume', default=False, action='store_true', help='Resume training (default: %(default)s)')
    group_train.add_argument('--checkpoint', default=None, help='File name to save weights at checkpoint')
    group_train.add_argument('--freq', type=int, default=25, help='Report progress and save checkpoint every FREQ iteration (default: %(default)s)')
    group_train.add_argument('--init', choices = ['gaussian', 'uniform'], default='gaussian', help='Initializion for weights (default: %(default)s)')

    group_postprocess = parser.add_argument_group('postprocess', 'Parameters for generating distance matrix')
    group_postprocess.add_argument('--load', default=Path(__file__).stem, help='File name to load weights (default: %(default)s)')
    group_postprocess.add_argument('--distances', default='distances', help='File name to save distance matrices (default: %(default)s)')

    group_extract = parser.add_argument_group('extract', 'Parameters for generating distance matrix')
    group_extract.add_argument('--triplets', default='triplets', help='File name to save distance matrices (default: %(default)s)')

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

def generate_sentences(doc,format='ZippedXml',n=None,log_file=stderr):
    corpus = Corpus.create(doc,format=format)
    sentence = []
    for word,tag in corpus.generate_tags(n,log_file=log_file):
        if tag.isalpha():
            sentence.append(word.lower())
        elif tag=='.':
            yield sentence
            sentence = []

def build_vocabulary(args,rng):
    with open(args.logfile,'w') as logfile:
        docnames = [doc for pattern in args.docnames for doc in glob(join(args.data,pattern))]
        vocabulary = Vocabulary()
        for doc in docnames:
            corpus = Corpus.create(doc,format=args.format)
            for word,tag in corpus.generate_tags(args.n,log_file=logfile):
                if tag.isalpha():
                    vocabulary.add(word.lower())

        vocabulary_file = create_file_name(args.vocabulary,path=args.data)
        vocabulary.save(vocabulary_file,docnames=docnames)
    print (f'Saved vocabulary of {len(vocabulary)} words to {vocabulary_file}')

def list_vocabulary(args,_):
    '''
    List items in vocabulary
    '''
    vocabulary = Vocabulary()
    vocabulary_file = create_file_name(args.vocabulary,path=args.data)
    vocabulary.load(vocabulary_file)
    word = Index2Word(vocabulary)
    for i,freq in vocabulary.items():
        try:
            print (word[i],freq)
        except UnicodeEncodeError as err:
            print (err)

def create_training_examples(args,rng):
    '''
    create training examples from corpus
    '''
    vocabulary = Vocabulary()
    vocabulary_file = create_file_name(args.vocabulary,path=args.data)
    vocabulary.load(vocabulary_file)

    docnames = [doc for pattern in args.docnames for doc in glob(join(args.data,pattern))]
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
        m = 1
        for doc in docnames:
            corpus = Corpus.create(doc,format=args.format)
            for sentence in corpus.generate_sentences(args.n):
                indices = vocabulary.parse(sentence)
                for word,context,y in word2vec.generate_examples([indices],tower):
                    examples.writerow([word,context,y])
                    m += 1
        print (f'Saved {m} examples to {examples_file}')

def train(args,rng):
    '''
    train weights using examples
    '''
    if len(args.docnames)>0:   # Issue 21
        exit(f'Docnames {args.docnames} with train does not make sense')
    k,width,paths,data = read_training_data(create_file_name(args.examples,ext='csv',path=args.data))
    model = Word2Vec()
    if args.resume:
        model.load(create_file_name(args.load,path=args.data))
    else:
        model.build(data[:,0].max()+1,n=args.dimension,rng=rng,init=args.init)
    loss_calculator = LossCalculator(model,data)
    checkpoint_file = None if args.checkpoint==None else create_file_name(args.checkpoint,path=args.data)
    optimizer = Optimizer.create(model,data,loss_calculator,
                                 m=args.minibatch,N=args.N,eta=args.eta,final_ratio=args.ratio,
                                 tau=establish_tau(args.tau,N=args.N,m=args.minibatch,M=len(data)),rng=rng,
                                 checkpoint_file=checkpoint_file,freq=args.freq)
    eta,total_loss = optimizer.optimize(is_stopping=is_stopping)
    model.save(create_file_name(args.save,path=args.data), width=width,k=k,paths=paths,total_loss=total_loss,eta=eta)

    fig = figure()
    ax1 = fig.add_subplot(1,1,1)
    x_scale = [args.freq*i for i in range(len(optimizer.log))]
    ax1.plot(x_scale,optimizer.log,color='xkcd:red')
    ax1.ticklabel_format(style='plain',axis='x',useOffset=False)
    ax1.set_title(f'{args.examples}/{args.vocabulary} Minibatch={args.minibatch}, dimension={model.n}')
    ax1.set_xlabel('step')
    ax1.set_ylabel('Loss',color='xkcd:red')
    ax1.set_ylim(bottom=0)
    ax2 = ax1.twinx()
    ax2.plot(x_scale,optimizer.etas,color='xkcd:blue')
    ax2.set_ylabel(r'$\eta$',color='xkcd:blue')
    ax2.set_ylim(bottom=0)
    fig.savefig(join(args.figs,args.plot))

def postprocess(args,rng):
    '''
    compute matrix of distances between vectors
    '''
    model = Word2Vec()
    model_name = create_file_name(args.load,path=args.data)
    model.load(model_name)
    vocabulary = Vocabulary()
    vocabulary_file = create_file_name(args.vocabulary,path=args.data)
    vocabulary.load(vocabulary_file)
    words = Index2Word(vocabulary)
    CosineDistances = np.abs(model.create_products())
    m,_ = CosineDistances.shape
    distances_name = create_file_name(args.distances,path=args.data)
    np.savez(distances_name,CosineDistances=CosineDistances)
    print (f'Saved distances in {distances_name}')
    fig = figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(CosineDistances.flatten(),bins=100)
    ax.set_title(f'Cosine Distances from {args.load}')
    fig.savefig(join(args.figs,args.plot))

def extract(args,rng):
    '''
     extract pairs of words, starting with closest
     '''
    def generate_pairs(Distances):
        '''
        Used to iterate through an array of distances, starting with the two closest points

        Parameters:
            Distances

        Yields:
           A sequence of pairs {...,(i1,j1), (i2,j2), ...} such that
           i1<j2, i2<j2 and Distances(i1,j1)<= Distances(i2,j2)
        '''
        m,n = Distances.shape
        indices = np.argsort(Distances,axis=None)   # Indices into flattened array in desired sequence
        Is,Js = np.unravel_index(indices, Distances.shape)
        for i,j in zip(Is,Js):
            if i<j:
                yield i,j

    vocabulary = Vocabulary()
    vocabulary_file = create_file_name(args.vocabulary,path=args.data)
    vocabulary.load(vocabulary_file)
    words = Index2Word(vocabulary)
    distances_name = create_file_name(args.distances,path=args.data)
    with np.load(distances_name) as data:
        CosineDistances = data['CosineDistances']
        print (f'Loaded {distances_name}')
    triplets_file = create_file_name(args.triplets,ext='csv',path=args.data)
    with open(triplets_file,'w', newline='') as out:
        triplets = writer(out)
        for i,j in generate_pairs(CosineDistances):
            try:
                triplets.writerow ([words[i],words[j],CosineDistances[i,j]])
            except UnicodeEncodeError as err:
                print (i, j, err)
    print (f'Saved triplets in {triplets_file}')

Commands = {
    'build'       : build_vocabulary,
    'create'      : create_training_examples,
    'train'       : train,
    'postprocess' : postprocess,
    'extract'     : extract,
    'list'        : list_vocabulary
}

if __name__=='__main__':
    rcParams['text.usetex'] = True
    start = time()
    args = create_arguments()
    Commands[args.command](args,default_rng(args.seed))
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
