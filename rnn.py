# Sean Robertsons's NLP demo: Classifying Names with a Character-Level RNN
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

class Alphabet:
    def __init__(self):
        self.all_letters = ascii_letters + " .,;'"
        self.n           = len(self.all_letters)

    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self,s):
        return ''.join(c for c in normalize('NFD', s) if category(c) != 'Mn' and c in self.all_letters)

    def letterToIndex(self,letter):
        return self.all_letters.find(letter)

    # Turn a line into a <line_length x 1 x n>,
    # or an array of one-hot letter vectors
    def lineToTensor(self,line):
        tensor = zeros(len(line), 1, self.n)
        for li, letter in enumerate(line):
            tensor[li][0][self.letterToIndex(letter)] = 1
        return tensor

class Categories:
    def __init__(self):
        self.category_lines = {}
        self.all_categories = []

    def load(self,pathname,alphabet):
        for filename in glob(pathname):
            self.add(filename,alphabet)

    def add(self,filename,alphabet):
        self.all_categories.append(splitext(basename(filename))[0])
        self.category_lines[self.all_categories[-1]] =  readLines(filename,alphabet)

    def get_n(self):
        return len(self.all_categories)

    def fromOutput(self,output):
        top_n, top_i = output.topk(1)
        category_i   = top_i[0].item()
        return self.all_categories[category_i], category_i

    # Get a random category and random line from that category
    def get_random_pair(self):
        category = randomChoice(self.all_categories)
        line     = randomChoice(self.category_lines[category])
        return category, line

    def get_random(self,alphabet):
        category,line   = self.get_random_pair()
        category_tensor = tensor([self.all_categories.index(category)], dtype=long)
        line_tensor     = alphabet.lineToTensor(line)
        return category, line, category_tensor, line_tensor

    def get_index(self,name):
        return self.all_categories.index(name)

class Timer:
    def __init__(self):
        self.start = time()

    def since(self):
        s   = time() - self.start
        m   = floor(s / 60)
        s  -= m * 60
        return (m, s)

# Read a file and split into lines
def readLines(filename,alphabet):
    lines  = open(filename, encoding='utf-8').read().strip().split('\n')
    return [alphabet.unicodeToAscii(line) for line in lines]

def randomChoice(l):
    return l[randint(0, len(l) - 1)]


def train(model,category_tensor, line_tensor, criterion, learning_rate=0.001):
    hidden = model.initHidden()
    model.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

def evaluate(model, line_tensor):
    hidden = model.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    return output

if __name__=='__main__':

    # Hyperparameters
    LEARNING_RATE   = 0.005
    N               = 100000
    PRINT_FREQUENCY = 5000
    PLOT_FREQUENCY  = 1000
    N_CONFUSION     = 10000
    N_HIDDEN        = 128
    DECAY_RATE      = 0.0

    alphabet        = Alphabet()
    categories      = Categories()
    timer           = Timer()
    criterion       = NLLLoss()

    for filename in glob('data/names/*.txt'):
        categories.add(filename,alphabet)

    rnn           = RNN(alphabet.n, N_HIDDEN, categories.get_n())
    current_loss  = 0
    all_losses    = []
    learning_rate = LEARNING_RATE
    for i in range(1, N + 1):
        category, line, category_tensor, line_tensor = categories.get_random(alphabet)
        output, loss                                 = train(rnn, category_tensor, line_tensor, criterion, learning_rate=learning_rate)
        current_loss                                += loss

        if i % PRINT_FREQUENCY == 0:
            guess, guess_i = categories.fromOutput(output)
            correct = '✓' if guess == category else f'✗ ({category})'
            m,s     = timer.since()
            print (f'{i}, {int((i / N) * 100)}%, {m}m {s:.0f}s, {loss:.4f}, {line}, {guess}, {correct}')

        if i % PLOT_FREQUENCY == 0:
            all_losses.append(current_loss / PLOT_FREQUENCY)
            current_loss = 0

        learning_rate *= (1 - DECAY_RATE)

    # Keep track of correct guesses in a confusion matrix
    confusion   = zeros(categories.get_n(), categories.get_n())

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(N_CONFUSION):
        category, line, category_tensor, line_tensor        = categories.get_random(alphabet)
        output                                              = evaluate(rnn, line_tensor)
        guess, guess_i                                      = categories.fromOutput(output)
        confusion[categories.get_index(category)][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(categories.get_n()):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = figure(figsize=(20,20))
    ax1 = fig.add_subplot(211)
    ax1.plot(all_losses)

    ax2 = fig.add_subplot(212)
    cax = ax2.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax2.set_xticklabels([''] + categories.all_categories, rotation=90)
    ax2.set_yticklabels([''] + categories.all_categories)

    # Force label at every tick
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.yaxis.set_major_locator(MultipleLocator(1))

    show()
