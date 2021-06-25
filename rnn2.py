# -*- coding: iso-8859-15 -*-

# https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
from __future__ import unicode_literals, print_function, division

from glob              import glob
from io                import open
from matplotlib.pyplot import figure, plot, show
from os.path  import basename, splitext
from random import randint
from rnn import Timer
import unicodedata
import string
from torch import cat, zeros, LongTensor
from torch.nn import Dropout, Linear, LogSoftmax, Module, NLLLoss

class RNN(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h         = Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o         = Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o         = Linear(hidden_size + output_size, output_size)
        self.dropout     = Dropout(0.1)
        self.softmax     = LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined  = cat((category, input, hidden), 1)
        hidden          = self.i2h(input_combined)
        output          = self.i2o(input_combined)
        output_combined = cat((hidden, output), 1)
        output          = self.o2o(output_combined)
        output          = self.dropout(output)
        output          = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return zeros(1, self.hidden_size)


# Random item from a list
def randomChoice(l):
    return l[randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line



def findFiles(path): return glob(path)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    with open(filename, encoding='utf-8') as some_file:
        return [unicodeToAscii(line.strip()) for line in some_file]

# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return LongTensor(letter_indexes)

# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor

def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size(0)

if __name__=='__main__':
    all_letters = string.ascii_letters + " .,;'-"
    n_letters = len(all_letters) + 1 # Plus EOS marker
    # Build the category_lines dictionary, a list of lines per category
    category_lines = {}
    all_categories = []
    for filename in findFiles('data/names/*.txt'):
        category = splitext(basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)

    if n_categories == 0:
        raise RuntimeError('Data not found. Make sure that you downloaded data '
            'from https://download.pytorch.org/tutorial/data.zip and extract it to '
            'the current directory.')

    print('# categories:', n_categories, all_categories)
    print(unicodeToAscii("O'Néàl"))

    criterion = NLLLoss()

    learning_rate = 0.0005

    timer = Timer()

    rnn = RNN(n_letters, 128, n_letters)

    n_iters = 100000
    print_every = 5000
    plot_every = 500
    all_losses = []
    total_loss = 0 # Reset every plot_every iters

    for iter in range(1, n_iters + 1):
        output, loss = train(*randomTrainingExample())
        total_loss += loss

        if iter % print_every == 0:
            m,s     = timer.since()
            print (f'{m} {s:.0f} {iter}, {iter / n_iters * 100:0f}, {loss}')

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0



    figure()
    plot(all_losses)
    show()
