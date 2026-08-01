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

'''Continuous Bag Of Words'''

from argparse import ArgumentParser
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import one_hot

class Corpus:
    def __init__(self,text = 'the quick brown fox jumped over the lazy dog'):
        self.text = text.split()
        self.words = list(set(self.text))
        self.indices = {self.words[i]:i for i in range(len(self.words))}
        self.tokenized = np.array([self.indices[word] for word in self.text])
        
    def get_vocabulary_size(self):
        return len(self.words)    
        
class WordData:
    def __init__(self,corpus,n):
        self.corpus = corpus
        self.n = n
        
    def __len__(self):
        return len(self.corpus.text) - self.n
    
    def __getitem__(self, idx):
        run = self.corpus.tokenized[idx:idx+self.n+1]
        context = np.delete(run,self.n//2)
        word = int(run[self.n//2])
        label = torch.Tensor.float(one_hot(torch.tensor(word),num_classes=len(self.corpus.words)))
        return context,label
    
    
class WordEmbeddings(nn.Module):
    def __init__(self,V,D,n):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=V*n, embedding_dim=D)
        self.linear_1 = nn.Linear(in_features=D, out_features=V)

    def forward(self, x):
        x = self.embeddings(x)
        x = x.mean(axis=1)
        return self.linear_1(x)
    
def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-D','--embedding_dim',type=int,default=300)
    parser.add_argument('-n','--windows_size',type=int,default=4)
    parser.add_argument('-N','--NIterators',type=int,default=100)
    parser.add_argument('--batch',type=int, default=4)
    return parser.parse_args()

def train(model,dataloader,NIterators=100):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.95)
    loss_fn = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'        
    for epoch in range(NIterators):
        train_loss = 0
        model.train()
        
        for feature, label in dataloader:
            model = model.to(device)
            y_train_pred = model(feature.to(device))
            loss = loss_fn(y_train_pred, label.to(device))
            train_loss = train_loss + loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_train_loss = train_loss / len(dataloader)
        print(f'Epoch:{epoch} | Mean Training Loss : {mean_train_loss}')  
        
def main():
    start  = time()
    args = parse_args()
    corpus = Corpus()
    training_data = WordData(corpus,args.windows_size)
    model = WordEmbeddings(corpus.get_vocabulary_size(),args.embedding_dim,args.windows_size)
 
    train_dataloader = DataLoader(training_data, batch_size=args.batch, shuffle=True)
    train(model,train_dataloader,NIterators=args.NIterators)
    

        
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    
if __name__=='__main__':
    main()
