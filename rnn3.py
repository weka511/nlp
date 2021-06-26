# Sean Robertsons's NLP demo: Translation with a Sequence to Sequence Network and Attention
# https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html

from __future__          import unicode_literals, print_function, division
from argparse            import ArgumentParser
from io                  import open
from matplotlib.pyplot   import figure, matshow, plot, show, subplots
from matplotlib.ticker   import MultipleLocator
from random              import choice, random
from re                  import sub
from rnn                 import Timer
from torch               import bmm, cat, cuda, device, long, no_grad, tensor, zeros
from torch.nn            import Dropout, Embedding, GRU, Linear, LogSoftmax, Module, NLLLoss
from torch.nn.functional import log_softmax, relu, softmax
from torch.optim         import SGD
from unicodedata         import category, normalize

class Language:
    SOS_token = 0
    EOS_token = 1
    def __init__(self, name):
        self.name       = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {Language.SOS_token: 'SOS',
                           Language.EOS_token: 'EOS'
                           }
        self.n_words    = len(self.index2word)  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word]         = self.n_words
            self.word2count[word]         = 1
            self.index2word[self.n_words] = word
            self.n_words                 += 1
        else:
            self.word2count[word]        += 1

class Encoder(Module):
    def __init__(self, input_size, hidden_size=256):
        super().__init__()
        self.device      = device('cuda' if cuda.is_available() else 'cpu')
        self.hidden_size = hidden_size
        self.embedding   = Embedding(input_size, hidden_size)
        self.gru         = GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded       = self.embedding(input).view(1, 1, -1)
        output         = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return zeros(1, 1, self.hidden_size, device=self.device)

class Decoder(Module):
    def __init__(self, hidden_size=256, output_size=256):
        super().__init__()
        self.device      = device('cuda' if cuda.is_available() else 'cpu')
        self.hidden_size = hidden_size
        self.embedding   = Embedding(output_size, hidden_size)
        self.gru         = GRU(hidden_size, hidden_size)
        self.out         = Linear(hidden_size, output_size)
        self.softmax     = LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output         = self.embedding(input).view(1, 1, -1)
        output         = relu(output)
        output, hidden = self.gru(output, hidden)
        output          = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return zeros(1, 1, self.hidden_size, device=self.device)


class AttentionDecoder(Module):
    def __init__(self, hidden_size=256, output_size=256, dropout_p=0.1, max_length=10):
        super().__init__()
        self.device       = device('cuda' if cuda.is_available() else 'cpu')
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
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(c for c in normalize('NFD', s) if category(c) != 'Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = sub(r'([.!?])', r' \1', s)
    s = sub(r'[^a-zA-Z.!?]+', r' ', s)
    return s

def readLanguages(lang1, lang2, reverse=False):
    file_name = 'data/%s-%s.txt' % (lang1, lang2)
    print(f'Reading lines {file_name}')

    # Read the file and split into lines
    lines = open(file_name, encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs       = [list(reversed(p)) for p in pairs]
        input_lang  = Language(lang2)
        output_lang = Language(lang1)
    else:
        input_lang =  Language(lang1)
        output_lang = Language(lang2)

    return input_lang, output_lang, pairs



eng_prefixes = (
    'i am ',    'i m ',
    'he is',    'he s ',
    'she is',   'she s ',
    'you are',  'you re ',
    'we are',   'we re ',
    'they are', 'they re '
)


def filterPair(p,MAX_LENGTH = 10):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLanguages(lang1, lang2, reverse)
    print(f'Read {len(pairs)} sentence pairs')
    pairs = filterPairs(pairs)
    print(f'Trimmed to {len(pairs)} sentence pairs')
    print('Counting words...')
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print('Counted words:')
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(Language.EOS_token)
    return tensor(indexes, dtype=long, device='cpu').view(-1, 1)  # FIXME


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)




def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length            = 10,
          teacher_forcing_ratio = 0.5,
          device                = 'cpu'):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length    = input_tensor.size(0)
    target_length   = target_tensor.size(0)
    encoder_outputs = zeros(max_length, encoder.hidden_size, device = device)
    loss            = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input       = tensor([[Language.SOS_token]], device=device)
    decoder_hidden      = encoder_hidden
    use_teacher_forcing = random() < teacher_forcing_ratio

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi                                        = decoder_output.topk(1)
            decoder_input                                     = topi.squeeze().detach()  # detach from history as input
            loss                                             += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == Language.EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def showPlot(points):
    figure()
    fig, ax = subplots()
    # this locator puts ticks at regular intervals
    loc = MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plot(points)

def trainIters(encoder, decoder, N, print_every=1000, plot_every=100, learning_rate=0.01):
    timer             = Timer()
    plot_losses       = []
    print_loss_total  = 0  # Reset every print_every
    plot_loss_total   = 0  # Reset every plot_every
    encoder_optimizer = SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = SGD(decoder.parameters(), lr=learning_rate)
    training_pairs    = [tensorsFromPair(choice(pairs)) for i in range(N)]
    criterion = NLLLoss()

    for i in range(1, N + 1):
        training_pair     = training_pairs[i - 1]
        input_tensor      = training_pair[0]
        target_tensor     = training_pair[1]
        loss              = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total  += loss

        if i % print_every == 0:
            print_loss_avg   = print_loss_total / print_every
            print_loss_total = 0
            m,s              = timer.since()
            print (f'{m}m {s:.0f}s, {i}, {i / N * 100:.0f}%, {print_loss_avg}')

        if i % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

def evaluate(encoder, decoder, sentence, max_length=10,device='cpu'):
    with no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = tensor([[Language.SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == Language.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = figure()
    ax  = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

def evaluateAndShowAttention(input_sentence,encoder, decoder):
    output_words, attentions = evaluate(encoder, decoder, input_sentence)
    print(f'input = {input_sentence}')
    print('output = {" ".join(output_words)}')
    showAttention(input_sentence, output_words, attentions)

if __name__ =='__main__':
    parser = ArgumentParser('Translation with a Sequence to Sequence Network and Attention')
    parser.add_argument('--N',         type = int,   default = 75000)
    parser.add_argument('--Frequency', type = int,   default = 5000)
    parser.add_argument('--Plot',      type = int,   default = 100)
    parser.add_argument('--hidden',    type = int,   default = 256)
    parser.add_argument('--lr',        type = float, default = 0.01)
    parser.add_argument('--dropout',   type = float, default = 0.01)
    args                           = parser.parse_args()

    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    encoder                        = Encoder(input_lang.n_words, hidden_size=args.hidden)#.to(device)
    decoder                        = AttentionDecoder(hidden_size=args.hidden, output_size=output_lang.n_words, dropout_p=args.dropout)#.to(device)

    trainIters(encoder, decoder, args.N, print_every=args.Frequency,plot_every=args.Plot,learning_rate=args.lr)
    # evaluateRandomly(encoder, decoder)

    evaluateAndShowAttention('elle a cinq ans de moins que moi .',encoder, decoder)

    evaluateAndShowAttention('elle est trop petit .',encoder, decoder)

    evaluateAndShowAttention('je ne crains pas de mourir .',encoder, decoder)

    evaluateAndShowAttention('c est un jeune directeur plein de talent .',encoder, decoder)

    show()
