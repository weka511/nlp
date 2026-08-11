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
    
class LogSoftmax:
    
    def log_softmax(self,z):
        return z - np.log(np.exp(z).sum())
    
    def _call__(self,prediction,label):
        '''
        Parameters:
            prediction
            label
        '''
    
if __name__=='__main__':
        main()