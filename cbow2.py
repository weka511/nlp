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
Train a Continuous Bag of Words Model
'''

from pickle import dump, load
from queue import Queue
from unittest import main,TestCase
import numpy as np
from numpy.testing import assert_array_equal,assert_array_almost_equal

class OneHotFactory:
    '''
    Encodes tokens as 1-hot vectors for use in neural network
    '''
    def __init__(self,n:int = 19):
        self.n = n
        
    def create(self,tokens:[int]):
        try:
            one_hot = np.zeros(tokens.shape + (self.n,))
            for i in range(len(tokens)):
                one_hot[i,tokens[i]] = 1
        except AttributeError:
            one_hot = np.zeros(self.n)
            one_hot[tokens] = 1 
        return one_hot

class Model:
    '''
    CBOW neural network
    '''
    @staticmethod
    def create(file_name,encoder=None):
        '''
        Load a model that has previously been saved
        
        Parameters:
            file_name    Path to file 
        '''
        with open(file_name, 'rb') as file:
            loaded_data = load(file)
            m,n = loaded_data['P'].shape
            product = Model(m=m,n=n,encoder=encoder)
            product.P = loaded_data['P']
            product.H = loaded_data['H']
            return product
        
    def __init__(self,m=19,n=300,rng = np.random.default_rng(),encoder=None):
        self.P = rng.uniform(low=-1,high=+1,size=(m,n))
        self.H = rng.uniform(low=-1,high=+1,size=(n,m))
        self.encoder = encoder
        
    def get_average(self,w):
        '''
        Used to average the vectors making up the context
        
        Returns:
            A single word vector
        '''
        return np.average(w,axis=0)
        
    def forward(self,X):
        '''
        Given an input to projection layer, fill in projection and hidden layers
        '''
        self.X = X
        self.y = np.dot(X,self.P)
        z = np.dot(self.y,self.H)
        return z
    
    def __call__(self,feature:[int]):
        '''
        Used to average a context and pass it through the network.
        '''
        W = np.array([self.encoder.create(w) for w in feature])
        X = self.get_average(W)        
        return self.forward(X)
    
    def save(self,file_name):
        '''
        Save model in a file
        
        Parameters:
            file_name    Path to file 
        '''
        with open(file_name,'wb') as out:
            dump({
                'P':self.P,
                'H':self.H
                },
                 out)

class CrossEntropyLoss:
    '''
    Used to calculate cross entropy loss
    
    See A Brief Overview of Cross Entropy Loss, Chris Hughes,
    https://medium.com/@chris.p.hughes10/a-brief-overview-of-cross-entropy-loss-523aa56b75d5
    '''
    def __init__(self,model=None):
        self.model = model
        
    def log_softmax(self,z):
        self.exp_z = np.exp(z)
        return z - np.log(self.exp_z.sum())
    
    def log_safe_softmax(self,z):
        '''
        Numerically stable version Goodfellow et al, eq (6.33)
        '''
        return self.log_softmax(z - z.max())
    
    def _cross_entropy(self,log_probabilities,label):
        return -np.dot(log_probabilities,label).sum()
    
    def __call__(self,prediction,label):
        '''
        Parameters:
            prediction
            label        A int
        '''
        self.label = label
        return self._cross_entropy(self.log_safe_softmax(prediction),label)
    
    def backward(self):
        dLoss_dz = -self.label + self.exp_z
        self.dLoss_dH = np.outer(dLoss_dz,self.model.y).T
        self.dLoss_dP = np.outer(np.matmul(self.model.H,dLoss_dz),self.model.X).T


class GradientDescent:
    def __init__(self,model,loss_fn,lr=0.01):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        
    def step(self):
        self.model.H -= self.lr*self.loss_fn.dLoss_dH
        self.model.P -= self.lr*self.loss_fn.dLoss_dP
            
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
        
class TestOneHotFactory(TestCase):
    def setUp(self):
        self.factory = OneHotFactory(n=7)  
        
    def test_factory(self):
        assert_array_equal(np.array([[0,1,0,0,0,0,0],
                                     [0,0,1,0,0,0,0],
                                     [0,0,0,0,1,0,0],
                                     [0,0,0,0,0,1,0]]),
                           self.factory.create(np.array([1,2,4,5])))
        
class TestModel(TestCase):
    def setUp(self):
        self.corpus = NanoCorpus()
        self.factory = OneHotFactory(n=self.corpus.get_n()+1)
        self.model = Model(m=self.corpus.get_n(),n=11)
       
    def test_convert_1hot(self):
        '''
        Verify calulcation of a gaggle of 1-hot vectors from a sequence of tokens
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