#    Copyright (C) 2021 Simon A. Crase
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

# https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb

from argparse            import ArgumentParser
from itertools           import chain
from matplotlib.pyplot   import figure, legend, plot, savefig, show, title, xlabel, ylabel
from numpy               import array
from numpy.random        import default_rng
from sys                 import float_info
from torch               import dot, flip, from_numpy, load, matmul, norm, randn,  save, zeros
from torch.autograd      import Variable
from torch.nn.functional import log_softmax, nll_loss

# tokenize_corpus

def tokenize_corpus(corpus):
    return [x.split() for x in corpus]

# create_vocabulary

def create_vocabulary(tokenized_corpus):
    vocabulary = {token for sentence in tokenized_corpus for token in sentence}
    return vocabulary,                                     \
           {w: idx for (idx, w) in enumerate(vocabulary)}, \
           {idx: w for (idx, w) in enumerate(vocabulary)}

# create_idx_pairs
#
# Create list of pairs, (word,context)

def create_idx_pairs(tokenized_corpus, word2idx,
                     window_size = 2):

    idx_pairs = []

    for sentence in tokenized_corpus:
        indices = [word2idx[word] for word in sentence]

        for center_word_pos in range(len(indices)):
            for word_offset in chain(range(-window_size,0), range(1, window_size + 1)):
                context_word_pos = center_word_pos + word_offset
                if context_word_pos >= 0 and context_word_pos < len(indices):
                    context_word_idx = indices[context_word_pos]
                    idx_pairs.append((indices[center_word_pos], context_word_idx))

    return array(idx_pairs)

# get_input_layer

def get_input_layer(word_idx,vocabulary_size):
    x           = zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x

# shuffled
#
# Generator for shuffling idx_pairs

def shuffled(idx_pairs, rg = None):
    indices = list(range(len(idx_pairs)))
    if rg!=None:
        rg.shuffle(indices)
    for index in indices:
        yield idx_pairs[index]

# train

def train(idx_pairs,vocabulary_size,
          lr             = 0.01,
          decay_rate     = 0,
          burn_in        = 0,
          num_epochs     = 1000,
          embedding_dims = 5,
          frequency      = 100,
          alpha          = 0.9,
          rg             = None):
    W1     = Variable(randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
    W2     = Variable(randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
    Delta1 = zeros(embedding_dims, vocabulary_size)
    Delta2 = zeros(vocabulary_size, embedding_dims)
    Losses = []
    Epochs = []

    print (f'Decay rate={decay_rate}')
    for epoch in range(num_epochs):
        loss_val      = 0
        learning_rate = lr/(1+decay_rate * epoch)

        for data, target in shuffled(idx_pairs,rg):
            x           = Variable(get_input_layer(data,vocabulary_size)).float()
            y_true      = Variable(from_numpy(array([target])).long())
            z1          = matmul(W1, x)
            z2          = matmul(W2, z1)
            y_predicted = log_softmax(z2, dim=0)
            loss        = nll_loss(y_predicted.view(1,-1), y_true)
            loss_val   += loss.item()
            loss.backward()
            Delta1     = alpha * Delta1 - learning_rate * W1.grad.data
            Delta2     = alpha * Delta2 - learning_rate * W2.grad.data
            W2.data    += Delta2
            W1.data    += Delta1
            W1.grad.data.zero_()
            W2.grad.data.zero_()

        if epoch % frequency == 0:
            print(f'Loss at epoch {epoch}: {loss_val/len(idx_pairs)}')
            if epoch > burn_in:
                Epochs.append(epoch)
                Losses.append(loss_val/len(idx_pairs))

    return W1,W2,Epochs,Losses

def get_similarity(v,u):
    return dot(v,u)/(norm(v)*norm(u))

# compare

# https://gist.github.com/mbednarski/da08eb297304f7a66a3840e857e060a0#gistcomment-3689982

def compare(word1,word2,
            W1              = None,
            vocabulary_size = 0,
            word2idx        = {}):


    s = get_similarity(matmul(W1,get_input_layer(word2idx[word1],vocabulary_size)),
                       matmul(W1,get_input_layer(word2idx[word2],vocabulary_size)))
    print (f'{word1} {word2} {s}')
    return s

# corpus

def corpus(file_name):
    with open(file_name) as f:
        for line in f:
            yield line.strip('.\n')



if __name__=='__main__':
    parser = ArgumentParser('Build word2vector')
    parser.add_argument('action',      choices=['train', 'test'],                          help = 'Train weights or test them')
    parser.add_argument('--N',         type = int,   default = 20001,                      help = 'Number of Epochs for training')
    parser.add_argument('--lr',        type = float, default = 0.01,                       help = 'Learning rate (before decay)')
    parser.add_argument('--alpha',     type = float, default = 0.9,                        help = 'Momentum')
    parser.add_argument('--decay',     type = float, default = [0.01], nargs='+',          help = 'Decay rate for learning')
    parser.add_argument('--frequency', type = int,   default = 100,                        help = 'Frequency for display')
    parser.add_argument('--n',         type = int,   default = 2,                          help = 'Window size')
    parser.add_argument('--m',         type = int,   default = 5,                          help ='Embedding size')
    parser.add_argument('--output',                  default = 'out',                      help = 'Output file name')
    parser.add_argument('--burn',      type=int,     default = None,                       help = 'Burn in')
    parser.add_argument('--show',                    default = False, action='store_true', help='Show plots')
    parser.add_argument('--shuffle',                 default = False, action='store_true', help='Shiffle indices before each epoch')
    parser.add_argument('--corpus',                  default = 'nano-corpus.txt',          help = 'Corpus file name')
    args = parser.parse_args()

    if args.action == 'train':
        tokenized_corpus             = tokenize_corpus(corpus(args.corpus))
        vocabulary,word2idx,idx2word = create_vocabulary(tokenized_corpus)
        vocabulary_size              = len(vocabulary)
        idx_pairs                    = create_idx_pairs(tokenized_corpus, word2idx,
                                                        window_size = args.n)

        figure(figsize=(10,10))

        minimum_loss = float_info.max

        for decay_rate in args.decay:
            W1,W2,Epochs,Losses = train(idx_pairs,vocabulary_size,
                                        lr             = args.lr,
                                        decay_rate     = decay_rate,
                                        burn_in        = 2*args.frequency if args.burn ==None else args.burn,
                                        num_epochs     = args.N,
                                        embedding_dims = args.m,
                                        frequency      = args.frequency,
                                        alpha          = args.alpha,
                                        rg             = default_rng() if args.shuffle else None)
            plot(Epochs,Losses,label=f'Decay rate={decay_rate}')

            if Losses[-1]<minimum_loss:
                minimum_loss = Losses[-1]
                print (f'Saving weights for Loss={minimum_loss} in {args.output}.pt')
                save ({'W1'         : W1,
                       'W2'         : W2,
                       'word2idx'   : word2idx,
                       'idx2word'   : idx2word,
                       'decay_rate' : decay_rate,
                       'idx_pairs'  : idx_pairs},
                      f'{args.output}.pt')

        xlabel('Epoch')
        ylabel('Loss')
        legend()
        title(f'{args.corpus} -- Embedding dimensions={args.m}, momentum={args.alpha}')
        savefig(args.output)

    if args.action == 'test':
        loaded            = load(f'{args.output}.pt')
        W1                = loaded['W1']
        W2                = loaded['W2']
        word2idx          = loaded['word2idx']
        idx2word          = loaded['idx2word']
        idx_pairs         = loaded['idx_pairs']

        _,vocabulary_size = W1.shape

        for idx, word in idx2word.items():
            word_vector      = W1[:,idx]
            sims             = matmul(word_vector,W1)
            most_similar_ids = flip(sims.argsort(),[0])
            sim_words        = [idx2word[i] for i in most_similar_ids.tolist()]
            print (f'{word}\t{" ".join([s for s in sim_words])} {"Shuffled" if args.shuffle else ""}')

    if args.show:
        show()
