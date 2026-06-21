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

''' Sean Robertsons's NLP demo: Generating Names with a Character-Level RNN
   https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
'''

from __future__        import unicode_literals, print_function, division
from matplotlib.pyplot import figure, plot, show
from rnn               import Alphabet, Categories, Timer
from torch             import cat, zeros, LongTensor, no_grad
from torch.nn          import Dropout, Linear, LogSoftmax, Module, NLLLoss

# RNN
#
# Recurrent neural network for learning association between names and languages,
# and generating names

class RNN(Module):
    def __init__(self, input_size=None, hidden_size=None, output_size=None,n_categories=None,dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h         = Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o         = Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o         = Linear(hidden_size + output_size, output_size)
        self.dropout     = Dropout(dropout)
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


# categoryTensor
#
# Create 1-hot vector for category
def categoryTensor(category):
    i            = categories.get_index(category)
    tensor       = zeros(1, categories.get_n())
    tensor[0][i] = 1
    return tensor

# inputTensor
#
# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = zeros(len(line), 1, alphabet.n)
    for i in range(len(line)):
        tensor[i][0][alphabet.get_index(line[i])] = 1

    return tensor

# targetTensor
#
# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [alphabet.get_index(line[i]) for i in range(1, len(line))]
    letter_indexes.append(alphabet.n - 1) # EOS
    return LongTensor(letter_indexes)

# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line     = categories.get_random_pair()
    category_tensor    = categoryTensor(category)
    input_line_tensor  = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor

# step
#
# Take one training step
#
# For each timestep (that is, for each letter in a training word) the inputs of the network will be
# (category, current letter, hidden state) and the outputs will be (next letter, next hidden state).
# So for each training set, we'll need the category, a set of input letters, and a set of output/target letters.
#
# Since we are predicting the next letter from the current letter for each timestep, the letter pairs
# are groups of consecutive letters from the line - e.g. for 'ABCD<EOS>' we would create ('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'EOS').

def step(category_tensor, input_line_tensor, target_line_tensor):
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

# sample
#
# Sample from a category and starting letter
def sample(category, start_letter='A',max_length = 20):
    with no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category)
        input           = inputTensor(start_letter)
        hidden          = rnn.initHidden()
        output_name     = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi     = output.topk(1)
            topi           = topi[0][0]
            if topi == alphabet.n - 1:
                break
            else:
                letter = alphabet.all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name

# get_samples
#
# Get multiple samples from one category and multiple starting letters
def get_samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))

if __name__=='__main__':
    alphabet        = Alphabet()
    categories      = Categories()
    categories.load('data/names/*.txt',alphabet)
    timer           = Timer()
    criterion       = NLLLoss()
    learning_rate   = 0.0005
    N               = 100000
    print_every     = 5000
    plot_every      = 500
    all_losses      = []
    total_loss      = 0
    rnn             = RNN(input_size   = alphabet.n,
                          hidden_size  = 128,
                          output_size  = alphabet.n,
                          n_categories = categories.get_n())
    for i in range(1, N + 1):
        output, loss = step(*randomTrainingExample())
        total_loss += loss

        if i % print_every == 0:
            m,s     = timer.since()
            print (f'{m} {s:.0f} {i}, {i / N * 100:.0f}%, {loss}')

        if i % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    get_samples('Russian', 'RUS')
    get_samples('German', 'GER')
    get_samples('Spanish', 'SPA')
    get_samples('Chinese', 'CHI')

    figure()
    plot(all_losses)
    show()
