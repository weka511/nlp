# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

from __future__        import unicode_literals, print_function, division
from glob              import glob
from io                import open
from math              import floor
from matplotlib.pyplot import figure, plot, show
from matplotlib.ticker import MultipleLocator
from os.path           import basename, splitext
from random            import randint
from string            import ascii_letters
from time              import time
from torch             import cat, long, tensor, zeros
from torch.nn          import Module, Linear, LogSoftmax, NLLLoss
from unicodedata       import normalize,category


class RNN(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h         = Linear(input_size + hidden_size, hidden_size)
        self.i2o         = Linear(input_size + hidden_size, output_size)
        self.softmax     = LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = cat((input, hidden), 1)
        hidden   = self.i2h(combined)
        output   = self.i2o(combined)
        output   = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return zeros(1, self.hidden_size)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(c for c in normalize('NFD', s) if category(c) != 'Mn' and c in all_letters)

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i   = top_i[0].item()
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[randint(0, len(l) - 1)]

def randomTrainingExample():
    category        = randomChoice(all_categories)
    line            = randomChoice(category_lines[category])
    category_tensor = tensor([all_categories.index(category)], dtype=long)
    line_tensor     = lineToTensor(line)
    return category, line, category_tensor, line_tensor

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

def timeSince(since):
    now = time()
    s   = now - since
    m   = floor(s / 60)
    s  -= m * 60
    return '%dm %ds' % (m, s)

def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

if __name__=='__main__':
    all_letters    = ascii_letters + " .,;'"
    n_letters      = len(all_letters)
    category_lines = {}
    all_categories = []

    for filename in glob('data/names/*.txt'):
        all_categories.append(splitext(basename(filename))[0])
        category_lines[all_categories[-1]] =  readLines(filename)

    n_categories  = len(all_categories)
    n_hidden      = 128
    rnn           = RNN(n_letters, n_hidden, n_categories)
    criterion     = NLLLoss()
    learning_rate = 0.005
    n_iters       = 100000
    print_every   = 5000
    plot_every    = 1000
    current_loss  = 0 # Keep track of losses for plotting
    all_losses   = []
    start        = time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss                                 = train(category_tensor, line_tensor)
        current_loss                                += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else f'✗ ({category})'
            print (f'{iter}, {int((iter / n_iters) * 100)}%, {timeSince(start)}, {loss:.4f}, {line}, {guess}, {correct}')

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    # Keep track of correct guesses in a confusion matrix
    confusion   = zeros(n_categories, n_categories)
    n_confusion = 10000

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output                                       = evaluate(line_tensor)
        guess, guess_i                               = categoryFromOutput(output)
        category_i                                   = all_categories.index(category)
        confusion[category_i][guess_i]              += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = figure(figsize=(20,20))
    ax1 = fig.add_subplot(211)
    ax1.plot(all_losses)

    ax2 = fig.add_subplot(212)
    cax = ax2.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax2.set_xticklabels([''] + all_categories, rotation=90)
    ax2.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.yaxis.set_major_locator(MultipleLocator(1))

    show()
