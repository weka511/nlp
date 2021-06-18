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
#
#  This program has been written to test my understanding of word2vec -- https://arxiv.org/abs/1301.3781/
#
# The code is based on Mateusz Bednarski's article, Implementing word2vec in PyTorch (skip-gram model)
# https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb

from argparse            import ArgumentParser
from glob                import glob
from itertools           import chain
from matplotlib.pyplot   import figure, legend, plot, savefig, show, title, xlabel, ylabel
from numpy               import array
from numpy.random        import default_rng
from os                  import remove
from re                  import compile
from sys                 import float_info
from time                import time
from tokenizer           import extract_sentences, extract_tokens, read_text
from torch               import dot, flip, from_numpy, load, matmul, norm, randn,  save, zeros
from torch.autograd      import Variable
from torch.nn.functional import log_softmax, nll_loss

# tokenize_corpus
#
# Convert corpus (list of lists of words) to list of tokens, each being a words in lower case.
#
# Parameters:
#      corpus  (list of lists of words, punctuation, etc)
# Returns:
#      List of lists of tokens

def tokenize_corpus(corpus):
    def istoken(s):
        return s.isalpha()
    def lower(S):
        return [s.lower() for s in S if istoken(s)]
    return [lower(x.split()) for x in corpus]

# create_vocabulary
#
# Extract list of words from document.
#
# Parameters:
#      tokenized_corpus      List of lists of tokens from corpus
#
# Returns:
#     vocabulary    List of all words in corpus
#     word2idx      Map word to an index in vocabulary
#     idx2word      Map index to word

def create_vocabulary(tokenized_corpus):
    vocabulary = list({token for sentence in tokenized_corpus for token in sentence})
    return vocabulary,                                     \
           {w: idx for (idx, w) in enumerate(vocabulary)}, \
           {idx: w for (idx, w) in enumerate(vocabulary)}

# create_idx_pairs
#
# Create list of pairs, (word,context) for corpus
#
# Parameters:
#      tokenized_corpus      List of lists of tokens from corpus
#      word2idx              Map word to an index in vocabulary
#      window_size           size of sliding window - used to determine whether a neighbour of a word is in context

def create_idx_pairs(tokenized_corpus, word2idx, window_size = 2):
    # flatten
    #
    # Convert a list of lists to a list containing all the same data

    def flatten(Lists):
        return [x for List in Lists for x in List]

    # create_contexts
    #
    # Create list of pairs, (word,context) for a single sentence

    def create_contexts(sentence):
        # in_window
        #
        # Used to verify that a context word is within the window
        def in_window(context_word_pos):
            return context_word_pos >= 0 and context_word_pos < len(sentence)

        # create_context
        #
        # Create a word context pair from a pair of positions within sentence
        def create_context(center_word_pos, word_offset):
            return (word_indices[center_word_pos], word_indices[center_word_pos+ word_offset])

        word_indices = [word2idx[word] for word in sentence]
        pos_offset_pairs = [(center_word_pos, context_word_offset)
                            for center_word_pos in range(len(sentence))
                                for context_word_offset in chain(range(-window_size,0), range(1, window_size + 1))
                                    if in_window(center_word_pos + context_word_offset)]

        return [create_context(center_word_pos, context_word_offset) for center_word_pos, context_word_offset in pos_offset_pairs]

    return flatten([create_contexts(sentence) for sentence in tokenized_corpus])



# create-1hot-vector
#
# Create input data for neural net - a 1-hot vector
#
# Parameters:
#     word_idx             Index of word in vocabulary
#     vocabulary_size      Size of vocabulary
#
# Returns:
#       1 hot vector of dimension vocabulary_size

def create_1hot_vector(word_idx,vocabulary_size):
    x           = zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x

# shuffled
#
# Generator for shuffling idx_pairs
#
# Parameters:
#      idx_pairs
#      rg   Either a numpy.random.default_rng (shuffle)
#           or None                           (no shuffle)

def shuffled(idx_pairs, rg = None):
    indices = list(range(len(idx_pairs)))
    if rg!=None:
        rg.shuffle(indices)
    for index in indices:
        yield idx_pairs[index]

# train
#
# Train neural network

def train(W1              = Variable(randn(0,0).float(), requires_grad=True),
          W2              = Variable(randn(0,0).float(), requires_grad=True),
          idx_pairs       = [],
          vocabulary_size = 0,
          lr              = 0.01,
          decay_rate      = 0,
          burn            = 0,
          num_epochs      = 1000,
          embedding       = 5,
          frequency       = 100,
          alpha           = 0.9,
          shuffle         = False,
          checkpoint      = lambda Epochs,Losses: None):
    start  = time()
    rg     = default_rng() if shuffle else None
    Delta1 = zeros(embedding, vocabulary_size)
    Delta2 = zeros(vocabulary_size, embedding)
    Losses = []
    Epochs = []

    print (f'Decay rate={decay_rate}')
    for epoch in range(num_epochs):
        loss_val      = 0
        learning_rate = lr/(1+decay_rate * epoch)

        for data, target in shuffled(idx_pairs,rg):
            x           = Variable(create_1hot_vector(data,vocabulary_size)).float()
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
            print(f'Loss at epoch {epoch}: {loss_val/len(idx_pairs)}. Time/epoch = {(time()-start)/(epoch+1):.0f} seconds')
            if epoch >= burn:
                Epochs.append(epoch)
                Losses.append(loss_val/len(idx_pairs))
                checkpoint(Epochs,Losses)

    return W1,W2,Epochs,Losses

# get_similarity
#
# Determine similarity between two vctors

def get_similarity(v,u):
    return dot(v,u)/(norm(v)*norm(u))


# read_corpus
#
# Read corpous from file

def read_corpus(file_name):
    with open(file_name) as f:
        for line in f:
            yield line.strip('.\n')

# save_checkpoint
#
# Save checkpoint, and delete excess checkpoint files

def save_checkpoint(obj,
                    base            = 'CHK',
                    seq             = 0,
                    max_checkpoints = 3):
    save(obj,f'{base}{seq:06d}.pt')
    checkpoints = sorted(glob(f'{base}*.pt'), reverse = True)
    if len(checkpoints)>max_checkpoints:
        for file_name in checkpoints[max_checkpoints:]:
            remove(file_name)

# get_output
#
# Used to determine output file name

def get_output(output=None, saved=None, corpus=[]):
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

if __name__=='__main__':
    parser = ArgumentParser('Build word2vector')
    parser.add_argument('action', choices=['train',
                                           'test',
                                           'resume'],                                                help = 'Train weights or test them')
    parser.add_argument('--N',                   type = int,   default = 20001,                      help = 'Number of Epochs for training')
    parser.add_argument('--lr',                  type = float, default = 0.001,                      help = 'Learning rate (before decay)')
    parser.add_argument('--alpha',               type = float, default = 0.0,                        help = 'Momentum')
    parser.add_argument('--decay',               type = float, default = [0.01], nargs='+',          help = 'Decay rate for learning')
    parser.add_argument('--frequency',           type = int,   default = 1,                          help = 'Frequency for display')
    parser.add_argument('--window',              type = int,   default = 2,                          help = 'Window size')
    parser.add_argument('--embedding',           type = int,   default = 100,                        help = 'Embedding size')
    parser.add_argument('--output',                            default = None,                       help = 'Output file name (train or resume)')
    parser.add_argument('--saved',                             default = None,                       help = 'Saved weights (resume or test)')
    parser.add_argument('--burn',                type=int,     default = 0,                          help = 'Burn in')
    parser.add_argument('--show',                              default = False, action='store_true', help = 'Show plots')
    parser.add_argument('--shuffle',                           default = False, action='store_true', help = 'Shuffle indices before each epoch')
    parser.add_argument('--corpus',                            default = None,  nargs='+',           help = 'Corpus file name')
    parser.add_argument('--chk',                               default = 'chk',                      help = 'Base for checkpoint file name')
    parser.add_argument('--depth',               type = int,   default = 16,                         help = 'Number of matches to display when testingt')
    parser.add_argument('--max_checkpoints',     type = int,   default = 3,                          help = 'Maximum number of checkpoints to be retained')
    args = parser.parse_args()

    if args.action == 'train':
        output_file                  = get_output(output=args.output, corpus=args.corpus)
        tokenized_corpus             = [word for word in extract_sentences(extract_tokens(read_text(file_names=args.corpus)))]
        vocabulary,word2idx,idx2word = create_vocabulary(tokenized_corpus)
        vocabulary_size              = len(vocabulary)
        idx_pairs                    = create_idx_pairs(tokenized_corpus, word2idx, window_size = args.window)
        print (f'Vocabulary size={vocabulary_size:,d} words. There are {len(idx_pairs):,d} idx pairs')

        figure(figsize=(10,10))

        minimum_loss = float_info.max

        for decay_rate in args.decay:
            W1 = Variable(randn(args.embedding, vocabulary_size).float(), requires_grad=True)
            W2 = Variable(randn(vocabulary_size, args.embedding).float(), requires_grad=True)
            W1,W2,Epochs,Losses = train(
                                    W1              = W1,
                                    W2              = W2,
                                    idx_pairs       = idx_pairs,
                                    vocabulary_size = vocabulary_size,
                                    lr              = args.lr,
                                    decay_rate      = decay_rate,
                                    burn            = args.burn,
                                    num_epochs      = args.N,
                                    embedding       = args.embedding,
                                    frequency       = args.frequency,
                                    alpha           = args.alpha,
                                    shuffle         = args.shuffle,
                                    checkpoint      = lambda Epochs,Losses: save_checkpoint (
                                        {   'W1'         : W1,
                                            'W2'         : W2,
                                            'word2idx'   : word2idx,
                                            'idx2word'   : idx2word,
                                            'decay_rate' : args.decay,
                                            'idx_pairs'  : idx_pairs,
                                            'args'       : args,
                                            'Epochs'     : Epochs,
                                            'Losses'     : Losses},
                                       seq             = Epochs[-1],
                                       base            = args.chk,
                                       max_checkpoints = args.max_checkpoints))

            plot(Epochs,Losses,label=f'Decay rate={decay_rate}')

            if Losses[-1]<minimum_loss:
                minimum_loss = Losses[-1]
                print (f'Saving weights for Loss={minimum_loss} in {output_file}.pt')
                save (
                    {   'W1'         : W1,
                        'W2'         : W2,
                        'word2idx'   : word2idx,
                        'idx2word'   : idx2word,
                        'decay_rate' : decay_rate,
                        'idx_pairs'  : idx_pairs,
                        'args'       : args,
                        'Epochs'     : Epochs,
                        'Losses'     : Losses},
                    f'{output_file}.pt')

        xlabel('Epoch')
        ylabel('Loss')
        legend()
        title(f'{args.corpus} -- Embedding dimensions={args.embedding}, momentum={args.alpha}')
        savefig(args.output)

    if args.action == 'resume':
        output_file       = get_output(output=args.output, saved=args.saved)
        loaded            = load(f'{args.saved}.pt')
        W1                = loaded['W1']
        W2                = loaded['W2']
        word2idx          = loaded['word2idx']
        idx2word          = loaded['idx2word']
        idx_pairs         = loaded['idx_pairs']
        loaded_args       = loaded['args']
        _,vocabulary_size = W1.shape
        W1,W2,Epochs,Losses = train(W1              = W1,
                                    W2              = W2,
                                    idx_pairs       = idx_pairs,
                                    vocabulary_size = vocabulary_size,
                                    lr              = args.lr,
                                    decay_rate      = args.decay[0],
                                    burn            = -1,         # force all data to be stored
                                    num_epochs      = args.N,
                                    embedding       = loaded_args.embedding,
                                    frequency       = args.frequency,
                                    alpha           = args.alpha,
                                    shuffle         = loaded_args.shuffle,
                                    checkpoint      = lambda Epochs,Losses: save_checkpoint (
                                        {   'W1'         : W1,
                                            'W2'         : W2,
                                            'word2idx'   : word2idx,
                                            'idx2word'   : idx2word,
                                            'decay_rate' : args.decay,
                                            'idx_pairs'  : idx_pairs,
                                            'args'       : loaded_args,
                                            'Epochs'     : Epochs,
                                            'Losses'     : Losses},
                                       seq             = Epochs[-1],
                                       base            = args.chk,
                                       max_checkpoints = args.max_checkpoints))


        minimum_loss      = Losses[-1]
        loaded_args.alpha = args.alpha
        loaded_args.lr    = args.lr
        print (f'Saving weights for Loss={minimum_loss} in {output_file}.pt')
        save (
            {   'W1'         : W1,
                'W2'         : W2,
                'word2idx'   : word2idx,
                'idx2word'   : idx2word,
                'decay_rate' : args.decay,
                'idx_pairs'  : idx_pairs,
                'args'       : loaded_args,
                'Epochs'     : Epochs,
                'Losses'     : Losses},
            f'{output_file}.pt')

    if args.action == 'test':
        loaded            = load(f'{args.output}.pt')
        W1                = loaded['W1']
        W2                = loaded['W2']
        word2idx          = loaded['word2idx']
        idx2word          = loaded['idx2word']
        idx_pairs         = loaded['idx_pairs']

        _,vocabulary_size = W1.shape

        mismatches = []
        for idx, word in idx2word.items():
            word_vector      = W1[:,idx]
            sims             = matmul(word_vector,W1)/(norm(word_vector)*norm(W1))
            most_similar_ids = flip(sims.argsort(),[0]).tolist()[:args.depth]
            sim_words        = [(idx2word[i],sims[i]) for i in most_similar_ids]
            first_word,_     = sim_words[0]
            if word==first_word:
                print (f'{" ".join([f"{w}({sim:.3f})" for w,sim in sim_words])} {"Shuffled" if args.shuffle else ""}')
            else:
                mismatches.append((word,sim_words))

        if len(mismatches)>0:
            print ('There were mismatches')
            for word,sim_words in mismatches:
                print (f'{word}\t{" ".join([f"{w}({sim:.3f})" for w,sim in sim_words])} {"Shuffled" if args.shuffle else ""}')

    if args.show:
        show()
