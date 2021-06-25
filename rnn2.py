# https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
from __future__        import unicode_literals, print_function, division
from matplotlib.pyplot import figure, plot, show
from random            import randint
from rnn               import Alphabet, Categories, Timer
from torch             import cat, zeros, LongTensor, no_grad
from torch.nn          import Dropout, Linear, LogSoftmax, Module, NLLLoss

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


# Random item from a list
def randomChoice(l):
    return l[randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(categories.all_categories)
    line     = randomChoice(categories.category_lines[category])
    return category, line

# One-hot vector for category
def categoryTensor(category):
    li = categories.all_categories.index(category)
    tensor = zeros(1, categories.get_n())
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = zeros(len(line), 1, alphabet.n)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][alphabet.all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [alphabet.all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(alphabet.n - 1) # EOS
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

# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABC'):
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
        output, loss = train(*randomTrainingExample())
        total_loss += loss

        if i % print_every == 0:
            m,s     = timer.since()
            print (f'{m} {s:.0f} {i}, {i / N * 100:.0f}%, {loss}')

        if i % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    samples('Russian', 'RUS')
    samples('German', 'GER')
    samples('Spanish', 'SPA')
    samples('Chinese', 'CHI')

    figure()
    plot(all_losses)
    show()
