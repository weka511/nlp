#!/usr/bin/env python

#    Copyright (C) 2021-2023 Simon A. Crase   simon@greenweaves.nz
#
#    This is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This software is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>
#
'''  This program has been written to test my understanding of word2vec --
   Efficient Estimation of Word Representations in Vector Space -- Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean--
   https://arxiv.org/abs/1301.3781/

   The code is based on Mateusz Bednarski's article, Implementing word2vec in PyTorch (skip-gram model)
   https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb
'''

from argparse            import ArgumentParser
from glob                import glob
from icecream            import ic
from itertools           import chain
from os                  import remove
from random              import sample
from re                  import compile
from sys                 import float_info
from time                import time

from matplotlib.pyplot   import figure, show
import numpy             as np
from numpy.random        import default_rng
import torch
from torch.autograd      import Variable
from torch.nn            import Module
from torch.nn.functional import log_softmax, nll_loss

from tokenizer           import extract_sentences, extract_tokens, read_text

class Word2Vec(Module):
    '''
    Word2Vec

    This class represents the two sets of weights described in Mateusz Bednarski's article
    '''
    def __init__(self,embedding_size, vocabulary_size):
        '''
        Initialize weights and gradients
        '''
        super().__init__()
        self.W1 = Variable(torch.randn(embedding_size, vocabulary_size).float(), requires_grad=True)
        self.W2 = Variable(torch.randn(vocabulary_size, embedding_size).float(), requires_grad=True)
        self.Delta1 = torch.zeros_like(self.W1)
        self.Delta2 = torch.zeros_like(self.W2)

    def calculate_loss(self,word_index,target,mult=True):
        '''
        calculate_loss

        Evaluate with one datum, compute loss, and update gradients
        '''
        y_true = Variable(torch.from_numpy(np.array([target])).long())
        if mult:
            x  = Variable(create_1hot_vector(word_index,vocabulary_size)).float()
            z1 = torch.matmul(self.W1, x)
        else:
            z1 = self.W1[:,word_index]
        z2 = torch.matmul(self.W2, z1)
        y_predicted = log_softmax(z2, dim=0)
        loss = nll_loss(y_predicted.view(1,-1), y_true)
        loss_val = loss.item()
        loss.backward()
        return loss_val

    def update(self,
             alpha = 0.9,
             lr    = 0.001):
        '''
        update

        Use accumulated gradients to update weights
        '''
        self.Delta1 = alpha*self.Delta1 - lr*self.W1.grad.data
        self.Delta2 = alpha*self.Delta2 - lr*self.W2.grad.data
        self.W1.data += self.Delta1
        self.W2.data += self.Delta2
        self.W1.grad.data.zero_()
        self.W2.grad.data.zero_()

    def get_weights(self):
        '''
        get_weights

        Return weights so they can be plotted
        '''
        w1    = torch.flatten(self.W1).detach().numpy()
        w2    = torch.flatten(self.W2).detach().numpy()
        return w1,w2


    def get_similarities(self,idx):
        '''
        get_similarities

        '''
        word_vector = self.W1[:,idx]
        return torch.matmul(word_vector,self.W1)/(torch.norm(word_vector)*torch.norm(self.W1))



class GradientDescent:
    '''
    GradientDescent

    Optimizer, used to train netwrok by Gradient Descent
    '''

    def __init__(self,
                 lr    = 0.001,
                 decay = 0,
                 rg    = None,
                 alpha = 0.9):
        self.lr = lr
        self.decay = decay
        self.rg = rg
        self.alpha = alpha


    def shuffled(self,idx_pairs):
        '''
        shuffled

        Generator for shuffling idx_pairs

        Parameters:
            idx_pairs
            rg   Either a numpy.random.default_rng (shuffle) or None
        '''
        indices = list(range(len(idx_pairs)))
        if self.rg!=None:
            self.rg.shuffle(indices)
        for index in indices:
            yield idx_pairs[index]

    def train(self,model,epoch,idx_pairs):
        loss_val = 0
        n        = 0
        lr       = self.lr/(1+self.decay * epoch)

        for word_index, target in self.shuffled(idx_pairs):
            loss_val += model.calculate_loss(word_index,target)
            n        += 1
            model.update(alpha = self.alpha, lr = self.lr)

        return loss_val/n



class StochasticGradient:
    '''
    StochasticGradient

    Optimizer, used to train network by Stochastic Gradient Descent
    '''
    def __init__(self,
                 lr    = 0.001,
                 decay = 0,
                 alpha = 0.9,
                 n     = 18):
        self.lr = lr
        self.decay = decay
        self.alpha = alpha
        self.n = n

    def train(self,model,epoch,idx_pairs):
        loss_val = 0
        lr       = self.lr/(1 + self.decay*epoch)

        for i in sample(range(len(idx_pairs)), self.n):
            word_index,target = idx_pairs[i]
            loss_val += model.calculate_loss(word_index,target)
        model.update(alpha = self.alpha, lr = self.lr)

        return loss_val/self.n


def tokenize_corpus(corpus):
    '''
    tokenize_corpus

    Convert corpus (list of lists of words) to list of tokens, each being a words in lower case.

     Parameters:
          corpus  (list of lists of words, punctuation, etc)
     Returns:
          List of lists of tokens
    '''
    def istoken(s):
        return s.isalpha()
    def lower(S):
        return [s.lower() for s in S if istoken(s)]
    return [lower(x.split()) for x in corpus]



def create_vocabulary(tokenized_corpus):
    '''
     create_vocabulary

    Extract list of words from document.

    Parameters:
         tokenized_corpus      List of lists of tokens from corpus

    Returns:
        vocabulary    List of all words in corpus
        word2idx      Map word to an index in vocabulary
        idx2word      Map index to word
    '''
    vocabulary = list({token for sentence in tokenized_corpus for token in sentence})
    return vocabulary,                                     \
           {w: idx for (idx, w) in enumerate(vocabulary)}, \
           {idx: w for (idx, w) in enumerate(vocabulary)}



def create_idx_pairs(tokenized_corpus, word2idx, window_size = 2):
    '''
     create_idx_pairs

     Create list of pairs, (word,context) for corpus

     Parameters:
          tokenized_corpus      List of lists of tokens from corpus
          word2idx              Map word to an index in vocabulary
          window_size           size of sliding window - used to determine whether a neighbour of a word is in context
    '''
    def flatten(Lists):
        '''
        flatten

        Convert a list of lists to a list containing all the same data
        '''
        return [x for List in Lists for x in List]


    def create_contexts(sentence):
        '''
        create_contexts

        Create list of pairs, (word,context) for a single sentence
        '''

        def in_window(context_word_pos):
            '''
            in_window

            Used to verify that a context word is within the window
            '''
            return context_word_pos >= 0 and context_word_pos < len(sentence)


        def create_context(center_word_pos, word_offset):
            '''
            create_context

         Create a word context pair from a pair of positions within sentence
            '''
            return (word_indices[center_word_pos], word_indices[center_word_pos+ word_offset])

        word_indices = [word2idx[word] for word in sentence]
        pos_offset_pairs = [(center_word_pos, context_word_offset)
                            for center_word_pos in range(len(sentence))
                                for context_word_offset in chain(range(-window_size,0), range(1, window_size + 1))
                                    if in_window(center_word_pos + context_word_offset)]

        return [create_context(center_word_pos, context_word_offset) for center_word_pos, context_word_offset in pos_offset_pairs]

    return flatten([create_contexts(sentence) for sentence in tokenized_corpus])


def create_1hot_vector(word_idx,vocabulary_size):
    '''
    create-1hot-vector

    Create input data for neural net - a 1-hot vector

    Parameters:
        word_idx             Index of word in vocabulary
        vocabulary_size      Size of vocabulary

    Returns:
          1 hot vector of dimension vocabulary_size
    '''
    x           = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x

def create_optimizer(name,
                     lr    = 0.01,
                     decay = 0,
                     shuffle = False,
                     alpha = 0.9,
                     n     = 8):
    '''
    create_optimizer

    Factory method used to set up either GradientDescent or StochasticGradient
    '''
    if name=='gradient':
        return GradientDescent(lr    = lr,
                               decay = decay,
                               rg    = default_rng() if shuffle else None,
                               alpha = alpha)
    if name=='stochastic':
        return StochasticGradient(lr    = lr,
                                  decay = decay,
                                  alpha = alpha,
                                  n     = n)



def train(model,
          idx_pairs       = [],
          vocabulary_size = 0,
          lr              = 0.01,
          decay           = 0,
          burn            = 0,
          num_epochs      = 1000,
          embedding       = 5,
          frequency       = 100,
          alpha           = 0.9,
          shuffle         = False,
          checkpoint      = lambda Epochs,Losses: None,
          optimizer_name  = 'gradient',
          n               = 8):
    '''
    train

    Train neural network
    '''
    start  = time()
    Losses = []
    Epochs = []

    optimizer = create_optimizer(optimizer_name,
                                 lr      = lr,
                                 decay   = decay,
                                 shuffle = shuffle,
                                 alpha   = alpha,
                                 n       = n)

    for epoch in range(num_epochs):
        mean_loss = optimizer.train(model,epoch,idx_pairs)

        if epoch % frequency == 0:
            print(f'Mean Loss at Epoch {epoch}: {mean_loss:.2f}. Time/epoch = {(time()-start)/(epoch+1):.3f} seconds')
            if epoch >= burn:
                Epochs.append(epoch)
                Losses.append(mean_loss)
                checkpoint(Epochs,Losses)

    return Epochs,Losses



def get_similarity(v,u):
    '''
    get_similarity

    Determine similarity between two vctors
    '''
    return torch.dot(v,u)/(torch.norm(v)*torch.norm(u))


def read_corpus(file_name):
    '''
    read_corpus

    Read corpus from file
    '''
    with open(file_name) as f:
        for line in f:
            yield line.strip('.\n')



def save_checkpoint(obj,
                    base            = 'CHK',
                    seq             = 0,
                    max_checkpoints = 3):
    '''
    save_checkpoint

    Save checkpoint, and delete excess checkpoint files
    '''
    torch.save(obj,f'{base}{seq:06d}.pt')
    checkpoints = sorted(glob(f'{base}*.pt'), reverse = True)
    if len(checkpoints)>max_checkpoints:
        for file_name in checkpoints[max_checkpoints:]:
            remove(file_name)



def get_output(output=None, saved=None, corpus=[]):
    '''
    get_output

    Used to determine output file name
    '''
    if output != None: return output
    if saved!= None:
        match = compile('(\D+)(\d*)').search(saved)
        if match:
            digits = match[2]
            seq = int(digits) if len(digits)>0 else -1
            return f'{match[1]}{seq+1}'
    else:
        head  = corpus[0]
        parts = head.split('.')
        return f'{parts[0]}-0'

def plot_weights(model):
    fig = figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    w1,w2 = model.get_weights()
    ax.hist(w1, 50,
         density = True,
         alpha   = 0.5,
         label   = 'W1')
    ax.hist(w2, 50,
         density = True,
         alpha   = 0.5,
         label   = 'W2')
    ax.legend()
    ax.set_xlabel('W1,W2')
    ax.set_title('Weights')
    fig.savefig(f'{args.output}-weights')

def plot_losses(Epochs,Losses,args):
    fig = figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.plot(Epochs,Losses)
    ax.set_xlabel('Epoch'),
    ax.set_ylabel('Loss')

    ax.set_title(f'{args.corpus} -- Embedding dimensions={args.embedding}, momentum={args.alpha},optimizer={args.optimizer}')
    fig.savefig(args.output)

if __name__=='__main__':
    parser = ArgumentParser(__doc__)
    parser.add_argument('action', choices=['train',
                                           'test',
                                           'resume'],
                                  help = 'Train weights or test them')
    parser.add_argument('--N',                   type = int,   default = 20001,                      help = 'Number of Epochs for training')
    parser.add_argument('--lr',                  type = float, default = 0.001,                      help = 'Learning rate (before decay)')
    parser.add_argument('--alpha',               type = float, default = 0.0,                        help = 'Momentum')
    parser.add_argument('--decay',               type = float, default = 0.0,                        help = 'Decay rate for learning')
    parser.add_argument('--frequency',           type = int,   default = 10,                         help = 'Frequency for display')
    parser.add_argument('--window',              type = int,   default = 2,                          help = 'Window size')
    parser.add_argument('--embedding',           type = int,   default = 100,                        help = 'Embedding size')
    parser.add_argument('--output',                            default = 'word2vec',                 help = 'Output file name (train or resume)')
    parser.add_argument('--saved',                             default = None,                       help = 'Saved weights (resume or test)')
    parser.add_argument('--burn',                type=int,     default = 0,                          help = 'Burn in')
    parser.add_argument('--show',                              default = False, action='store_true', help = 'Show plots')
    parser.add_argument('--shuffle',                           default = False, action='store_true', help = 'Shuffle indices before each epoch')
    parser.add_argument('--corpus',                            default = None,  nargs='+',           help = 'Corpus file name')
    parser.add_argument('--chk',                               default = 'chk',                      help = 'Base for checkpoint file name')
    parser.add_argument('--depth',               type = int,   default = 16,                         help = 'Number of matches to display when testing')
    parser.add_argument('--max_checkpoints',     type = int,   default = 3,                          help = 'Maximum number of checkpoints to be retained')
    parser.add_argument('--optimizer', choices =['gradient',
                                                 'stochastic'],
                                                 default='gradient')
    parser.add_argument('--minibatch',           type = int,  default = 8)
    args = parser.parse_args()

    if args.action == 'train':
        output_file                  = get_output(output=args.output, corpus=args.corpus)
        tokenized_corpus             = [word for word in extract_sentences(extract_tokens(read_text(file_names=args.corpus)))]
        vocabulary,word2idx,idx2word = create_vocabulary(tokenized_corpus)
        vocabulary_size              = len(vocabulary)
        idx_pairs                    = create_idx_pairs(tokenized_corpus, word2idx, window_size = args.window)
        print (f'Vocabulary size={vocabulary_size:,d} words. There are {len(idx_pairs):,d} idx pairs')

        minimum_loss = float_info.max

        model = Word2Vec(args.embedding, vocabulary_size)

        Epochs,Losses = train(model,
                              idx_pairs       = idx_pairs,
                              vocabulary_size = vocabulary_size,
                              lr              = args.lr,
                              decay           = args.decay,
                              burn            = args.burn,
                              num_epochs      = args.N,
                              embedding       = args.embedding,
                              frequency       = args.frequency,
                              alpha           = args.alpha,
                              shuffle         = args.shuffle,
                              checkpoint      = lambda Epochs,Losses: save_checkpoint (
                                  {   'model'           : model.state_dict(),
                                      'word2idx'        : word2idx,
                                      'idx2word'        : idx2word,
                                      'decay'      : args.decay,
                                      'idx_pairs'       : idx_pairs,
                                      'args'            : args,
                                      'Epochs'          : Epochs,
                                      'Losses'          : Losses,
                                      'vocabulary_size' : vocabulary_size },
                                 seq             = Epochs[-1],
                                 base            = args.chk,
                                 max_checkpoints = args.max_checkpoints),
                            optimizer_name  = args.optimizer,
                            n               = args.minibatch)

        plot_losses(Epochs,Losses,args)
        plot_weights(model)
        if Losses[-1]<minimum_loss:
            minimum_loss = Losses[-1]
            print (f'Saving weights for Loss={minimum_loss} in {output_file}.pt')

            torch.save (
                {   'model'           : model.state_dict(),
                    'word2idx'        : word2idx,
                    'idx2word'        : idx2word,
                    'decay'           : args.decay,
                    'idx_pairs'       : idx_pairs,
                    'args'            : args,
                    'Epochs'          : Epochs,
                    'Losses'          : Losses,
                    'vocabulary_size' : vocabulary_size  },
                f'{output_file}.pt')

    if args.action == 'resume':
        output_file     = get_output(output=args.output, saved=args.saved)
        loaded          = torch.load(f'{args.saved}.pt')
        state_dict      = model.state_dict()
        word2idx        = loaded['word2idx']
        idx2word        = loaded['idx2word']
        idx_pairs       = loaded['idx_pairs']
        loaded_args     = loaded['args']
        vocabulary_size = loaded['vocabulary_size']
        model           = Word2Vec(loaded_args.embedding, vocabulary_size)
        model.load_state_dict(loaded['model'])

        Epochs,Losses = train(model,
                              idx_pairs       = idx_pairs,
                              vocabulary_size = vocabulary_size,
                              lr              = args.lr,
                              decay      = args.decay[0],
                              burn            = -1,         # force all data to be stored
                              num_epochs      = args.N,
                              embedding       = loaded_args.embedding,
                              frequency       = args.frequency,
                              alpha           = args.alpha,
                              shuffle         = loaded_args.shuffle,
                              checkpoint      = lambda Epochs,Losses: save_checkpoint (
                                  {   'model'           : model.state_dict(),
                                      'word2idx'        : word2idx,
                                      'idx2word'        : idx2word,
                                      'decay'      : args.decay,
                                      'idx_pairs'       : idx_pairs,
                                      'args'            : loaded_args,
                                      'Epochs'          : Epochs,
                                      'Losses'          : Losses,
                                      'vocabulary_size' : vocabulary_size },
                                 seq             = Epochs[-1],
                                 base            = args.chk,
                                 max_checkpoints = args.max_checkpoints))


        minimum_loss      = Losses[-1]
        loaded_args.alpha = args.alpha
        loaded_args.lr    = args.lr
        print (f'Saving weights for Loss={minimum_loss} in {output_file}.pt')
        ic(model)
        save (
            {   'model'      : model.state_dict(),
                'word2idx'   : word2idx,
                'idx2word'   : idx2word,
                'decay' : args.decay,
                'idx_pairs'  : idx_pairs,
                'args'       : loaded_args,
                'Epochs'     : Epochs,
                'Losses'     : Losses,
                'vocabulary_size' : vocabulary_size},
            f'{output_file}.pt')

    if args.action == 'test':
        loaded          = torch.load(f'{args.saved}.pt')
        loaded_args     = loaded['args']
        word2idx        = loaded['word2idx']
        idx2word        = loaded['idx2word']
        idx_pairs       = loaded['idx_pairs']
        vocabulary_size = loaded['vocabulary_size']
        model           = Word2Vec(loaded_args.embedding, vocabulary_size)
        model.load_state_dict(loaded['model'])

        mismatches = []
        for idx, word in idx2word.items():
            similarities     = model.get_similarities(idx)
            most_similar_ids = torch.flip(similarities.argsort(),[0]).tolist()[:args.depth]
            similar_words    = [(idx2word[i],similarities[i]) for i in most_similar_ids]
            first_word,_     = similar_words[0]
            if word==first_word:
                print (f'{" ".join([f"{w}({sim:.3f})" for w,sim in similar_words])} {"Shuffled" if args.shuffle else ""}')
            else:
                mismatches.append((word,similar_words))

        if len(mismatches)>0:
            print ('There were mismatches')
            for word,similar_words in mismatches:
                print (f'{word}\t{" ".join([f"{w}({sim:.3f})" for w,sim in similar_words])} {"Shuffled" if args.shuffle else ""}')

    if args.show:
        show()
