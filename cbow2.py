#!/usr/bin/env python

#   Copyright (C) 2026 Simon Crase

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''
Continuous Bag of Words Model
Classes for building examples and training model
'''

from abc import ABC, abstractmethod
from csv import writer, QUOTE_MINIMAL
from pickle import dump, load
from queue import Queue
from typing import TextIO
from unittest import main,TestCase,skip
import numpy as np
from numpy.testing import assert_array_equal,assert_array_almost_equal
import nltk
from nltk.corpus.reader.bnc import BNCCorpusReader
from vocabulary import Vocabulary

__version__ = '1.1'
__author__ = 'Simon Crase'

class BNC(ABC):
    '''
    Read text from BNC corpus
    '''
    def __init__(self, root=r'./data/2553/download',component=r'\w*'):
        '''
        Connect to Baby BNC
        '''      
        nltk.data.path.append(root)
        
    def filenames(self):
        '''
        This generator is used to iterate through filenames
        '''
        for filename in self.bnc.fileids():
            yield filename

    def sentences(self, path, stem=False):
        '''
        This generator is used to iterate through sentences in a specified file
        
        Parameters:
            path   Full pathname for file
            stem   If true, then use word stems instead of word strings.
        '''
        for sentence in self.bnc.sents(fileids=path,stem=stem):
            yield sentence

class BNC_Baby(BNC):
    def __init__(self, root=r'./data/2553/download',component=r'\w*'):
        '''
        Connect to Baby BNC
        '''
        super().__init__(root)
        self.bnc = BNCCorpusReader(root=root+r'\Texts', fileids=component+r'/\w*\.xml')
        
class BNC_Full(BNC):        
    def __init__(self, root=r'./data/2554/download\Texts'):
        '''
        Connect to Full BNC
        ''' 
        super().__init___(root)
        nltk.data.path.append(r'./data\2554\download')
        self.bnc = BNCCorpusReader(root=root, fileids=r'[A-K]/\w*/\w*\.xml')   

class ExampleSet:
    '''
    This class holds the words and contexts for one sentence
    
    Attributes:
        vocabulary   Words used in set
        SOS          Token for start of sentence
        EOS          Token for end of sentence
        window_size  Determines how many context words will be included in each example
        count        Total number of examples
    '''

    def __init__(self, window_size : int =4, vocabulary:Vocabulary=Vocabulary()):
        '''
        Parameters:
            window_size
            vocabulary
        '''
        self.vocabulary = vocabulary
        self.SOS = self.vocabulary.tokenize('<SOS>')
        self.EOS = self.vocabulary.tokenize('<EOS>')
        self.window_size = window_size
        self.count = 0
        
    def __len__(self):
        '''
        The length is the total number of examples
        '''
        return self.count
        
    def build(self, sentence : [str], out_file : TextIO):
        '''
        Construct words and contexts for one sentence
        
        Parameters:
            sentence
            out_file
        '''
        self.__accumulate__(self.__tokenize__(sentence), out_file)

    def __tokenize__(self,sentence : [str]) -> [int]:
        '''
        Convert sentence from a list of words to a list of tokens
        
        Parameters:
            sentence
        '''
        return ([self.SOS] +
                [self.vocabulary.tokenize(word.lower() if word.isalpha() else '<UNK>') for word in sentence] +
                [self.EOS])
    
    def __accumulate__(self, tokens: [int], out_file:TextIO):
        '''
        Convert a sequence of tokens, representing one sentence, to 
        words and contexts
        
        Parameters:
            tokens      The sequence of tokens
            out_file    Pathe name for output file
        '''
        out = writer(out_file, delimiter=',', quotechar='|', quoting=QUOTE_MINIMAL)
        start = 0
        end = start + 2 * self.window_size + 1
        n_entries = len(tokens) - end + 1
        while end <= len(tokens):
            run = [tokens[i] for i in range(start, end)]
            words = run[self.window_size]
            context = run[:self.window_size] + run[self.window_size + 1:]
            out.writerow([words] + context)
            start += 1
            end += 1
            self.count += 1
            
class OneHotFactory:
    '''
    Encodes tokens as 1-hot vectors for use in neural network
    '''
    def __init__(self,n:int = 19):
        self.n = n
        
    def create(self,tokens:[int]):
        '''
        Create a matrix of 1-hot vectors from a list of tokens
        
        Parameters:
            tokens     List of integers, one for each toen
        '''
        try:
            product = np.zeros(tokens.shape + (self.n,))
            for i in range(len(tokens)):
                product[i,tokens[i]] = 1
        except AttributeError:
            product = np.zeros(self.n)
            product[tokens] = 1 
        return product

class Model:
    '''
    Neural network representing Continuous Bag of Words Model
    
    Attributes:
        P          Projection layer
        H          Hidden layer
        encoder    Used to convert integers to 1-hot vectors
    '''
    @staticmethod
    def create(path,encoder=None):
        '''
        Load a model that has previously been saved
        
        Parameters:
            path     Path to file 
            encoder  Used to convert integers to 1-hot vectors
        '''
        with open(path, 'rb') as file:
            loaded_data = load(file)
            m,dimensionality = loaded_data['P'].shape
            product = Model(m=m,dimensionality=dimensionality,encoder=encoder)
            product.P = loaded_data['P']
            product.H = loaded_data['H']
            return product
        
    def __init__(self,m=19,dimensionality=300,rng = np.random.default_rng(),encoder=None):
        '''
        Parameters:
            m                Size of inpput layer
            dimensionality   Dimensionality of space that we poject ipout onto
            rng              Random number generator
            encoder          Used to convert integers to 1-hot vectors
        '''
        self.P = rng.uniform(low=-1,high=+1,size=(m,dimensionality))
        self.H = rng.uniform(low=-1,high=+1,size=(dimensionality,m))
        self.encoder = encoder
        
    def get_average(self,w):
        '''
        Used to average the vectors making up the context
        
        Parameters:
            w   Vectors making up the context
        
        Returns:
            A single word vector
        '''
        return np.average(w,axis=0)
        
    def forward(self,X):
        '''
        Given an input to projection layer, fill in projection and hidden layers
        
        Parameters:
             X        Input to projection layer
        '''
        self.X = X
        self.y = np.dot(X,self.P)
        z = np.dot(self.y,self.H)
        return z
    
    def __call__(self,feature:[int]):
        '''
        Used to average a context and pass it through the network.
        
        Parameters:
            feature    Vector making up context
        '''
        W = np.array([self.encoder.create(w) for w in feature])
        X = self.get_average(W)        
        return self.forward(X)
    
    def save(self,path):
        '''
        Save model in a file
        
        Parameters:
            path    Path to file 
        '''
        with open(path,'wb') as out:
            dump({
                'P':self.P,
                'H':self.H
                },
                 out)

class Loss(ABC):
    '''
    Used to calculate loss and its derivatives
    
    Attributes:
        model      The model whose loss we are trying to minimize
    '''
    def __init__(self,model=None):
        self.model = model
        
    @abstractmethod
    def __call__(self,prediction,label):
        '''
        Calculate loss
        
        Parameters:
            prediction   Prediction from model 
            label        1-hot vector representing true value
        '''
        
    @abstractmethod
    def backward(self):
        '''
        Perform back propagation: calculate losses for Projection layer and Hidden layer
        '''
        
class CrossEntropyLoss(Loss):
    '''
    Used to calculate cross entropy loss
    
    See A Brief Overview of Cross Entropy Loss, Chris Hughes,
    https://medium.com/@chris.p.hughes10/a-brief-overview-of-cross-entropy-loss-523aa56b75d5
    
    Attributes:
        label     True value from examples
        exp_z     Cached value of exp(z) to expedite calculation of gradinent
    '''
    def __init__(self,model=None):
        super().__init__(model=model)
        
    def log_softmax(self,z):
        '''
        Calculate log of softmax
        
        Parameters:
             z      Predictions from model
        '''
        self.exp_z = np.exp(z)
        return z - np.log(self.exp_z.sum())
    
    def log_safe_softmax(self,z):
        '''
        Numerically stable version Goodfellow et al, eq (6.33)
        
        Parameters:
            z       Predictions from model
        '''
        return self.log_softmax(z - z.max())
    
    def _cross_entropy(self,log_p,y):
        '''
        Calculate cross entropy using the equation from Chris Hughes's aritcle
        
        Parameters:
            y      The true proabilities of eack class
            log_p  Logs of the probabilities estimated by the model
        '''
        return -np.dot(log_p,y)
    
    def __call__(self,prediction,label):
        '''
        Calculate the cross entropy loss
        
        Parameters:
            prediction   Prediction from model 
            label        1-hot vector representing true value
        '''
        self.label = label
        return self._cross_entropy(self.log_safe_softmax(prediction),label)
    
    def backward(self):
        '''
        Perform back propagation: calculate losses for Projection layer and Hidden layer
        
        See pages 8 and 9 of my NLP notebook and also the pdf generated from cbow.tex
        '''
        dLoss_dz = -self.label + self.exp_z/self.exp_z.sum()                       # Equation (7)
        self.dLoss_dH = np.outer(dLoss_dz,self.model.y).T                          # Equation (9)
        self.dLoss_dP = np.outer(np.matmul(self.model.H,dLoss_dz),self.model.X).T  # Equation (12)

class Optimizer(ABC):
    '''
    Attributes:
       model      The model whose loss we are trying to minimize
       loss_fn    The loss function we are trying to minimize
       lr         Learning rate
    '''
    def __init__(self,model,loss_fn,lr=0.01):
        '''
        Parameters:
            model      The model whose loss we are trying to minimize
            loss_fn    The loss function we are trying to minimize
            lr         Learning rate
        '''
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr 
        
    @abstractmethod
    def step(self):
        '''
        Perform one step of optimization
        '''
        
class GradientDescent(Optimizer):
    '''
    Optimizer used to perform Gradient Descent
    '''
    def __init__(self,model,loss_fn,lr=0.01):
        '''
        Parameters:
            model      The model whose loss we are trying to minimize
            loss_fn    The loss function we are trying to minimize
            lr         Learning rate
        '''
        super().__init__(model,loss_fn,lr=lr)
        
    def step(self):
        '''
        Perform one step of gradient descent: adjust H and P in model
        '''        
        self.model.H -= self.lr*self.loss_fn.dLoss_dH
        self.model.P -= self.lr*self.loss_fn.dLoss_dP
            

class TestOneHotFactory(TestCase):
    '''
    Tests to verify that OneHotFactory sets up vectors correctly
    '''
    def setUp(self):
        self.factory = OneHotFactory(n=7)  
        
    def test_factory(self):
        assert_array_equal(np.array([[0,1,0,0,0,0,0],
                                     [0,0,1,0,0,0,0],
                                     [0,0,0,0,1,0,0],
                                     [0,0,0,0,0,1,0]]),
                           self.factory.create(np.array([1,2,4,5])))
        
class TestModel(TestCase):
    class NanoCorpus:
        def __init__(self):
            self.data = [
                [0,  2,  3,  4,  5, 1], #     he is      a  king.
                [0,  6,  3,  4,  7, 1], #    she is      a  queen.
                [0,  2,  3,  4,  8, 1], #     he is      a  man.
                [0,  6,  3,  4,  9, 1], #    she is      a  woman.
                [0, 10,  3, 11, 12, 1], # warsaw is poland  capital.
                [0, 14,  3, 14, 12, 1], # berlin is germany capital
                [0, 15,  3, 16, 12, 1]  # paris is  france  capital.
            ]
        
        def get_n(self):
            return max(max(row) for row in self.data)
        
        def __getitem__(self,i):
            return self.data[i]
        
    def setUp(self):
        self.corpus = self.NanoCorpus()
        self.factory = OneHotFactory(n=self.corpus.get_n()+1)
        self.model = Model(m=self.corpus.get_n(),dimensionality=11)
       
    def test_convert_1hot(self):
        '''
        Verify calculation of a gaggle of 1-hot vectors from a sequence of tokens
        '''
        Expected = np.zeros((6,17))
        Expected[0,0] = 1
        Expected[1,2] = 1
        Expected[2,3] = 1
        Expected[3,4] = 1
        Expected[4,8] = 1
        Expected[5,1] = 1
        assert_array_equal(Expected, 
                           np.array([self.factory.create(w) for w in self.corpus[2]]))
   
    def test_calculate_x(self):
        '''
        Verify calculation of mean of 1-hot vectors
        '''
        W = np.array([self.factory.create(w) for w in self.corpus[2]])
        x = self.model.get_average(W)
        self.assertEqual(1/6,x[0])
        self.assertEqual(1/6,x[2])
        self.assertEqual(1/6,x[3])
        self.assertEqual(1/6,x[4])
        self.assertEqual(0,x[5])
        self.assertEqual(1/6,x[8])
        self.assertEqual(1/6,x[1])
                                    
class TestCrossEntropy(TestCase):
    '''
    Tests for the CrossEntropyLoss class
    '''
    
    def setUp(self):
        self.loss_fn = CrossEntropyLoss()
        
    def test_brief1(self):
        '''
        An example from Chris Hughes' article. He has rounded values to 3 decimal places
        '''
        self.assertAlmostEqual(0.357,
                            self.loss_fn._cross_entropy(
                                np.log(
                                    np.array([0.7,0.2,0.1])),
                                np.array([1,0,0])),
                            delta=0.001)
    def test_brief2(self):
        '''
        An second example from Chris Hughes' article.
        '''
        self.assertAlmostEqual(0.105,
                            self.loss_fn._cross_entropy(
                                np.log(
                                    np.array([0.9,0.05,0.05])),
                                np.array([1,0,0])),
                            delta=0.001) 
        
    def test_brief3(self):
        '''
        An third example from Chris Hughes' article.
        '''
        self.assertAlmostEqual(2.303,
                            self.loss_fn._cross_entropy(
                                np.log(
                                    np.array([0.1,0.8,0.1])),
                                np.array([1,0,0])),
                            delta=0.001)      
        
    def test_log_safe_softmax(self):
        '''
        Verify that log_safe_softmax gies the same answer as log_softmax
        '''
        assert_array_almost_equal(
            self.loss_fn.log_softmax(np.array([1,2,3,4])),
            self.loss_fn.log_safe_softmax(np.array([1,2,3,4]))
        )
        assert_array_almost_equal(
            self.loss_fn.log_softmax(np.array([0.9,0.05,0.025,0.025])),
            self.loss_fn.log_safe_softmax(np.array([0.9,0.05,0.025,0.025]))
        )    
            
if __name__=='__main__':
        main()