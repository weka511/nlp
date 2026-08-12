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

'''Train CBOW'''

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
            m = len(tokens)       
            one_hot = np.zeros((m,self.n))
            for i in range(m):
                one_hot[i,tokens[i]] = 1.0
        except TypeError:
            one_hot = np.zeros((self.n))
            one_hot[tokens] = 1.0
        return one_hot            

class Model:
    '''
    CBOW neural network
    '''
    def __init__(self,m=19,n=300,rng = np.random.default_rng(),encoder=None):
        self.P = rng.uniform(low=-1,high=+1,size=(m,n))
        self.H = rng.uniform(low=-1,high=+1,size=(n,m))
        self.encoder = encoder
        
    def forward(self,X):
        Projected = np.dot(X,self.P)
        Hidden = np.dot(Projected,self.H)
        return Hidden
    
    def __call__(self,feature=[2,3,5,7]):
        X = self.encoder.encode(feature)
        return self.forward(X)
    
class CrossEntropyLoss:
    '''
    Used to calculate cross entropy loss
    
    See A Brief Overview of Cross Entropy Loss, Chris Hughes,
    https://medium.com/@chris.p.hughes10/a-brief-overview-of-cross-entropy-loss-523aa56b75d5
    '''
    def log_softmax(self,z):
        return z - np.log(np.exp(z).sum())
    
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
            label
        '''
        return self._cross_entropy(self.log_softmax(prediction),label)

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