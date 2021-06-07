# https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb

from itertools           import chain
from matplotlib.pyplot   import figure, legend, plot, savefig, show, xlabel, ylabel
from numpy               import array
from torch               import from_numpy, matmul, randn, save, zeros
from torch.autograd      import Variable
from torch.nn.functional import log_softmax, nll_loss


def tokenize_corpus(corpus):
    return [x.split() for x in corpus]

def create_vocabulary(tokenized_corpus):
    vocabulary = {token for sentence in tokenized_corpus for token in sentence}
    return vocabulary, {w: idx for (idx, w) in enumerate(vocabulary)}, {idx: w for (idx, w) in enumerate(vocabulary)}

def create_idx_pairs(tokenized_corpus, word2idx,
                     window_size = 2):

    idx_pairs = []

    for sentence in tokenized_corpus:
        indices = [word2idx[word] for word in sentence]

        for center_word_pos in range(len(indices)):
            for w in chain(range(-window_size,0), range(1, window_size + 1)):
                context_word_pos = center_word_pos + w
                if context_word_pos >= 0 and context_word_pos < len(indices):
                    context_word_idx = indices[context_word_pos]
                    idx_pairs.append((indices[center_word_pos], context_word_idx))

    return array(idx_pairs)

def get_input_layer(word_idx,vocabulary_size):
    x = zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x

def train(idx_pairs,vocabulary_size,
          lr             = 0.01,
          decay_rate     = 0,
          burn_in        = 0,
          num_epochs     = 1000,
          embedding_dims = 5,
          frequency      = 100):
    W1     = Variable(randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
    W2     = Variable(randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
    Losses = []
    Epochs = []
    print (f'Decay rate={decay_rate}')
    for epoch in range(num_epochs):
        loss_val = 0
        learning_rate = lr/(1+decay_rate * epoch)
        for data, target in idx_pairs:
            x           = Variable(get_input_layer(data,vocabulary_size)).float()
            y_true      = Variable(from_numpy(array([target])).long())
            z1          = matmul(W1, x)
            z2          = matmul(W2, z1)
            y_predicted = log_softmax(z2, dim=0)
            loss        = nll_loss(y_predicted.view(1,-1), y_true)
            loss_val   += loss.item()
            loss.backward()
            W1.data    -= learning_rate * W1.grad.data
            W2.data    -= learning_rate * W2.grad.data
            W1.grad.data.zero_()
            W2.grad.data.zero_()

        if epoch % frequency == 0:
            print(f'Loss at epoch {epoch}: {loss_val/len(idx_pairs)}')
            if epoch > burn_in:
                Epochs.append(epoch)
                Losses.append(loss_val/len(idx_pairs))

    return W1,W2,Epochs,Losses

if __name__=='__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser('Build word2vector')
    parser.add_argument('action', choices=['train'])
    parser.add_argument('--N',         type = int,   default = 20001,              help = 'Number of Epochs for training')
    parser.add_argument('--lr',        type = float, default = 0.01,               help = 'Learning rate (before decay)')
    parser.add_argument('--decay',     type = float, default = [0.005], nargs='+', help = 'Decay rate for learning')
    parser.add_argument('--frequency', type = int,   default = 100,                help = 'Frequency for display')
    parser.add_argument('--n',         type = int,   default = 2,                  help = 'Window size')
    parser.add_argument('--m',         type = int,   default = 5,                  help ='Embedding size')
    parser.add_argument('--output',                  default = 'out',              help = 'Output file name')
    parser.add_argument('--burn',      type=int,     default = None,               help = 'Burn in')
    parser.add_argument('--show',                    default = False,              action='store_true')
    args = parser.parse_args()

    corpus = [
        'he is a king',
        'she is a queen',
        'he is a man',
        'she is a woman',
        'warsaw is poland capital',
        'berlin is germany capital',
        'paris is france capital',
    ]

    if args.action == 'train':
        tokenized_corpus             = tokenize_corpus(corpus)
        vocabulary,word2idx,idx2word = create_vocabulary(tokenized_corpus)
        vocabulary_size              = len(vocabulary)
        idx_pairs                    = create_idx_pairs(tokenized_corpus, word2idx,
                                                        window_size = args.n)

        figure(figsize=(10,10))

        for decay_rate in args.decay:
            W1,W2,Epochs,Losses = train(idx_pairs,vocabulary_size,
                                        lr             = args.lr,
                                        decay_rate     = decay_rate,
                                        burn_in        = 2*args.frequency if args.burn ==None else args.burn,
                                        num_epochs     = args.N,
                                        embedding_dims = args.m,
                                        frequency      = args.frequency)
            plot(Epochs,Losses,label=f'Decay rate={decay_rate}')
            save(W1,f'{args.output}-W1.pt')
            save(W2,f'{args.output}-W2.pt')

        xlabel('Epoch')
        ylabel('Loss')
        legend()
        savefig(args.output)

    if args.show:
        show()
