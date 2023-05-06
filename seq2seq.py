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


''' This code has been adapted from Sean Robertsons's NLP demo:
 Translation with a Sequence to Sequence Network and Attention
 https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
'''

from __future__          import unicode_literals, print_function, division
from argparse            import ArgumentParser
from hashlib             import sha1
from io                  import open
from matplotlib.pyplot   import figure, matshow, plot, savefig, show, title, subplots, xlabel, ylabel
from matplotlib.ticker   import FixedLocator, MaxNLocator
from random              import choice, random
from re                  import sub
from rnn                 import Timer
from torch               import bmm, cat, load, long, no_grad, save, tensor, zeros
from torch.nn            import Dropout, Embedding, GRU, Linear, LogSoftmax, Module, NLLLoss
from torch.nn.functional import log_softmax, relu, softmax
from torch.optim         import SGD
from unicodedata         import category, normalize

# Language
#
# A container for all words in corpus from one language, e.g. 'eng', 'fra',...
class Language:
    SOS_token = 0
    EOS_token = 1

    def __init__(self, name):
        self.name       = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            Language.SOS_token: 'SOS',
            Language.EOS_token: 'EOS'
        }
        self.n    = len(self.index2word)

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word]   = self.n
            self.word2count[word]   = 1
            self.index2word[self.n] = word
            self.n                 += 1
        else:
            self.word2count[word]  += 1

    def get_n(self):
        return self.n

    def get_index(self,word):
        return self.word2index[word]

    def get_word(self,index):
        return self.index2word[index]

    def tensorFromSentence(self, sentence):
        return tensor([self.get_index(word) for word in sentence.split(' ')] + [Language.EOS_token],
                      dtype  = long).view(-1, 1)

# Encoder
#
# The encoder of a seq2seq network is a RNN that outputs some value
# for every word from the input sentence. For every input word the
# encoder outputs a vector and a hidden state, and uses the hidden state for the next input word.
#
# see Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
# Cho et al https://arxiv.org/abs/1406.1078
class Encoder(Module):
    def __init__(self, input_size,
                 hidden_size = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding   = Embedding(input_size, hidden_size)
        self.gru         = GRU(hidden_size, hidden_size)       # see Cho et al and https://pytorch.org/docs/stable/generated/torch.nn.GRU.html

    def forward(self, input, hidden):
        embedded       = self.embedding(input).view(1, 1, -1)
        output         = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return zeros(1, 1, self.hidden_size)

# Decoder
#
# see Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
# Cho et al https://arxiv.org/abs/1406.1078
class Decoder(Module):
    def __init__(self,
                 hidden_size = 256,
                 output_size = 256):
        super().__init__()
        self.hidden_size  = hidden_size
        self.output_size  = output_size
        self.embedding    = Embedding(output_size, hidden_size)
        self.gru          = GRU(hidden_size, hidden_size)
        self.out          = Linear(hidden_size, output_size)
        self.softmax      = LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_outputs):               #FIXME
        output         = self.embedding(input).view(1, 1, -1)
        output         = relu(output)
        output, hidden = self.gru(output, hidden)
        output         = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return zeros(1, 1, self.hidden_size)

    def get_description(self):
        return f'Decoder hidden_size={self.hidden_size}, output_size={self.output_size}'

    # adapt
    #
    # Allow either encoder to be used with minimal disruption to existing code

    def adapt(self,decoded,train = True):
        decoder_output, decoder_hidden= decoded
        if train:
            return decoder_output, decoder_hidden
        else:
            return decoder_output, decoder_hidden, None

# AttentionDecoder

class AttentionDecoder(Module):
    def __init__(self,
                 hidden_size = 256,
                 output_size = 256,
                 dropout_p   = 0.1,
                 max_length  = 10):
        super().__init__()
        self.hidden_size  = hidden_size
        self.output_size  = output_size
        self.dropout_p    = dropout_p
        self.max_length   = max_length
        self.embedding    = Embedding(self.output_size, self.hidden_size)
        self.attn         = Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout      = Dropout(self.dropout_p)
        self.gru          = GRU(self.hidden_size, self.hidden_size)
        self.out          = Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded       = self.embedding(input).view(1, 1, -1)
        embedded       = self.dropout(embedded)
        attn_weights   = softmax(self.attn(cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied   = bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output         = cat((embedded[0], attn_applied[0]), 1)
        output         = self.attn_combine(output).unsqueeze(0)
        output         = relu(output)
        output, hidden = self.gru(output, hidden)
        output         = log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

    def get_description(self):
        return f'Attention hidden_size = {self.hidden_size}, '\
               f'output_size = {self.output_size}, '          \
               f'max_length = {self.max_length}, '           \
               f'dropout = {self.dropout_p}'

    # adapt
    #
    # Allow either encoder to be used with minimal disruption to existing code

    def adapt(self,decoded,train = True):
        decoder_output, decoder_hidden, decoder_attention = decoded
        if train:
            return decoder_output, decoder_hidden
        else:
            return decoder_output, decoder_hidden, decoder_attention

def unicodeToAscii(s):
    return ''.join(c for c in normalize('NFD', s) if category(c) != 'Mn')

# readLanguages
#
# Create language pair

def readLanguages(lang1, lang2, reverse=False):
    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427

    # Lowercase, trim, and remove non-letter characters
    def normalizeString(s):
        s = unicodeToAscii(s.lower().strip())
        s = sub(r'([.!?])', r' \1', s)
        s = sub(r'[^a-zA-Z.!?]+', r' ', s)
        return s

    file_name = f'data/{lang1}-{lang2}.txt'
    print(f'Reading lines {file_name}')

    # Read the file and split into lines
    lines = open(file_name, encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs           = [list(reversed(p)) for p in pairs]
        input_language  = Language(lang2)
        output_language = Language(lang1)
    else:
        input_language =  Language(lang1)
        output_language = Language(lang2)

    return input_language, output_language, pairs


# Since there are a lot of example sentences and we want to train something quickly,
# we'll trim the data set to only relatively short and simple sentences. Here the maximum length
# is 10 words (that includes ending punctuation) and we are filtering to sentences that translate
# to the form 'I am' or 'He is' etc. (accounting for apostrophes replaced earlier)

def filterPairs(pairs,
                max_length = 10):
    def filterPair(pair,
                   eng_prefixes = (
                       'i am ',    'i m ',
                       'he is',    'he s ',
                       'she is',   'she s ',
                       'you are',  'you re ',
                       'we are',   'we re ',
                       'they are', 'they re '
                   )               ):
        return len(pair[0].split(' ')) < max_length and \
               len(pair[1].split(' ')) < max_length and \
               pair[1].startswith(eng_prefixes)

    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False,  max_length = 10):
    input_language, output_language, pairs = readLanguages(lang1, lang2, reverse)
    print(f'Read {len(pairs)} sentence pairs')
    pairs = filterPairs(pairs, max_length = max_length)
    print(f'Trimmed to {len(pairs)} sentence pairs')
    print('Counting words...')
    for pair in pairs:
        input_language.addSentence(pair[0])
        output_language.addSentence(pair[1])
    print('Counted words:')
    print(input_language.name, input_language.get_n())
    print(output_language.name, output_language.get_n())
    return input_language, output_language, pairs


def tensorsFromPair(pair):
    input_tensor  = input_language.tensorFromSentence(pair[0])
    target_tensor = output_language.tensorFromSentence(pair[1])
    return (input_tensor, target_tensor)

def step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length            = 10,
          teacher_forcing_ratio = 0.5):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length    = input_tensor.size(0)
    target_length   = target_tensor.size(0)
    encoder_outputs = zeros(max_length, encoder.hidden_size)
    loss            = 0

    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
        encoder_outputs[i]             = encoder_output[0, 0]

    decoder_input       = tensor([[Language.SOS_token]])
    decoder_hidden      = encoder_hidden
    use_teacher_forcing = random() < teacher_forcing_ratio


    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for i in range(target_length):
            decoder_output, decoder_hidden = decoder.adapt(decoder(decoder_input, decoder_hidden, encoder_outputs))
            loss += criterion(decoder_output, target_tensor[i])
            decoder_input = target_tensor[i]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for i in range(target_length):
            decoder_output, decoder_hidden = decoder.adapt(decoder(decoder_input, decoder_hidden, encoder_outputs))
            topv, topi                                        = decoder_output.topk(1)
            decoder_input                                     = topi.squeeze().detach()  # detach from history as input
            loss                                             += criterion(decoder_output, target_tensor[i])
            if decoder_input.item() == Language.EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# plotLosses

def plotLosses(Epochs,Losses,
               output      = 'rnn',
               N           = 100000,
               description = None):

    fig, ax = subplots(figsize=(15,15))

    plot(Epochs,Losses)
    xlabel('Epoch')
    ylabel('Loss')
    title(f'N={N}: {description}')
    savefig(f'{output}.png')

def train(encoder, decoder, N,
               print_every   = 1000,
               plot_every    = 100,
               learning_rate = 0.01,
               output        = 'rnn'):
    timer             = Timer()
    Losses            = []
    Epochs            = []
    print_loss_total  = 0  # Reset every print_every
    plot_loss_total   = 0  # Reset every plot_every
    encoder_optimizer = SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = SGD(decoder.parameters(), lr=learning_rate)
    training_pairs    = [tensorsFromPair(choice(pairs)) for i in range(N)]
    criterion         = NLLLoss()

    for i in range(1, N + 1):
        training_pair     = training_pairs[i - 1]
        input_tensor      = training_pair[0]
        target_tensor     = training_pair[1]
        loss              = step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total  += loss

        if i % print_every == 0:
            print_loss_avg   = print_loss_total / print_every
            print_loss_total = 0
            m,s              = timer.since()
            print (f'{m}m {s:.0f}s, {i}, {i / N * 100:.0f}%, {print_loss_avg}')

        if i % plot_every == 0:
            Losses.append( plot_loss_total / plot_every)
            Epochs.append(i)
            plot_loss_total = 0

    plotLosses(Epochs,Losses,
             output      = output,
             N           = N,
             description = decoder.get_description())

def evaluate(encoder, decoder, sentence,
             max_length      = 10,
             input_language  = None,
             output_language = None):
    with no_grad():
        input_tensor    = input_language.tensorFromSentence(sentence)
        input_length    = input_tensor.size()[0]
        encoder_hidden  = encoder.initHidden()
        encoder_outputs = zeros(max_length, encoder.hidden_size)

        for i in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
            encoder_outputs[i] += encoder_output[0, 0]

        decoder_input      = tensor([[Language.SOS_token]])
        decoder_hidden     = encoder_hidden
        decoded_words      = []
        decoder_attentions = zeros(max_length, max_length)

        for i in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder.adapt(decoder(decoder_input, decoder_hidden, encoder_outputs),
                                                                      train=False)
            if decoder_attention!=None:
                decoder_attentions[i]                             = decoder_attention.data
            topv, topi                                        = decoder_output.data.topk(1)
            if topi.item() == Language.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_language.get_word(topi.item()))

            decoder_input = topi.squeeze().detach()

        if decoder_attention==None:
            return decoded_words, None
        else:
            return decoded_words, decoder_attentions[:i + 1]

def evaluateRandomly(encoder, decoder,
                     n               = 10,
                     input_language  = None,
                     output_language = None,
                     pairs           = []):
    for i in range(n):
        pair = choice(pairs)
        print(f'>{pair[0]}')
        print(f'={pair[1]}')
        output_words, attentions = evaluate(encoder, decoder, pair[0],
                                            input_language  = input_language,
                                            output_language = output_language)
        output_sentence          = ' '.join(output_words)
        print(f'<{output_sentence}\n', output_sentence)

# showAttention
#
# Show association between input and output words

def showAttention(input_sentence, output_words, attentions,
                  seq     = 666,
                  outfile = 'rnn'):
    # Set up figure with colorbar
    fig = figure()
    ax  = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    # see https://stackoverflow.com/questions/63723514/userwarning-fixedformatter-should-only-be-used-together-with-fixedlocator

    ax.xaxis.set_major_locator(MaxNLocator('auto'))
    xticks_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(FixedLocator(xticks_loc))

    ax.yaxis.set_major_locator(MaxNLocator('auto'))
    yticks_loc = ax.get_yticks().tolist()
    ax.yaxis.set_major_locator(FixedLocator(yticks_loc))

    title(input_sentence)
    savefig(f'{outfile}-{seq}.png')

# evaluateAndShowAttention

def evaluateAndShowAttention(input_sentence,encoder, decoder,
                             output          = 'rnn',
                             input_language  = None,
                             output_language = None):
    output_words, attentions = evaluate(encoder, decoder, input_sentence,
                                        input_language  = input_language,
                                        output_language = output_language)
    print(f'input = {input_sentence}')
    print(f'output = {" ".join(output_words)}')
    if attentions==None: return

    showAttention(input_sentence, output_words, attentions,
                  outfile = output,
                  seq     = sha1(input_sentence.encode('utf-8')).hexdigest())
# create_decoder
#
# Factory method for creating decoder

def create_decoder(decoder,
                   output_size = 0,
                   hidden_size = 0,
                   max_length  = 0,
                   dropout     = 0.1):
    if decoder=='attention':
        return AttentionDecoder(hidden_size = hidden_size,
                                output_size = output_size,
                                dropout_p   = dropout,
                                max_length  = max_length)
    if decoder=='simple':
        return Decoder(hidden_size = hidden_size,
                       output_size = output_size)

# load_model
#
# Load encode and decoder from saved file

def load_model(load_file):
    loaded             = load(load_file)
    pairs              = loaded['pairs']
    old_args           = loaded['args']
    input_language     = loaded['input_language']
    output_language    = loaded['output_language']
    encoder            = Encoder(input_language.get_n(), hidden_size = old_args.hidden)
    decoder            = create_decoder(old_args.decoder,
                                        output_size = output_language.get_n(),
                                        dropout     = args.dropout,
                                        max_length  = args.max_length,
                                        hidden_size = args.hidden)

    encoder.load_state_dict(loaded['encoder_state_dict'])
    decoder.load_state_dict(loaded['decoder_dict'])
    return encoder, decoder, input_language, output_language,pairs

if __name__ =='__main__':
    parser = ArgumentParser('Translation with a Sequence to Sequence Network and Attention')
    parser.add_argument('action',       choices = ['train', 'evaluate'])
    parser.add_argument('--decoder',    choices = ['simple',
                                                   'attention'],
                                        default = 'attention',
                                        help    = 'Specify whether Attention is to be used')
    parser.add_argument('--N',          type = int,            default = 75000, help ='Number of iterations while training')
    parser.add_argument('--printf',     type = int,            default = 5000,  help = 'Frequency for printing')
    parser.add_argument('--frequency',  type = int,            default = 100,   help = 'Frequency for plotting')
    parser.add_argument('--max_length', type = int,            default = 10,    help = 'Maximum length for Attention Decoder')
    parser.add_argument('--hidden',     type = int,            default = 256,   help = 'Number of hidden units')
    parser.add_argument('--lr',         type = float,          default = 0.01,  help = 'Learning Rate')
    parser.add_argument('--dropout',    type = float,          default = 0.1,   help = 'Dropout probability')
    parser.add_argument('--output',     type = str,            default = 'rnn', help = 'Base name for plotting (train)')
    parser.add_argument('--load',       type = str,            default = 'rnn', help = 'Base name for plotting (evaluate)')
    parser.add_argument('--show',       action = 'store_true', default = False, help = 'Show plots at end of run')
    args = parser.parse_args()

    if args.action=='train':
        input_language, output_language, pairs = prepareData('eng', 'fra',
                                                             reverse    = True,
                                                             max_length = args.max_length)
        encoder                                = Encoder(input_language.get_n(), hidden_size = args.hidden)
        decoder                                = create_decoder(args.decoder,
                                                                dropout     = args.dropout,
                                                                output_size = output_language.get_n(),
                                                                max_length  = args.max_length,
                                                                hidden_size = args.hidden)

        train(encoder, decoder, args.N,
              print_every   = args.printf,
              plot_every    = args.frequency,
              learning_rate = args.lr,
              output        = args.output)

        save(
            {
                'encoder_state_dict' : encoder.state_dict(),
                'decoder_dict'       : decoder.state_dict(),
                'args'               : args,
                'pairs'              : pairs,
                'input_language'     : input_language,
                'output_language'    : output_language
                },
            f'{args.output}.pt')

    if args.action=='evaluate':
        encoder, decoder, input_language, output_language, pairs = load_model(f'{args.load}.pt')

        evaluateRandomly(encoder, decoder,
                         input_language  = input_language,
                         output_language = output_language,
                         pairs           = pairs)

        for sentence in {
            'elle a cinq ans de moins que moi .',
            'elle est trop petit .',
            'je ne crains pas de mourir .',
            'c est un jeune directeur plein de talent .' }:
            evaluateAndShowAttention(sentence, encoder, decoder,
                                     output          = args.output,
                                     input_language  = input_language,
                                     output_language = output_language)

    if args.show:
        show()
