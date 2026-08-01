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

class WordData:
    def __len__(self):
        return 100
    def __getitem__(self, idx):
        return np.array([1,2,4,5]),torch.Tensor.float(one_hot(torch.tensor(3),num_classes=1000))
    
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
    parser.add_argument('-N','--NIterators',type=int,default=12)
    return parser.parse_args()
    
def main():
    start  = time()
    args = parse_args()
    model = WordEmbeddings(1000,args.embedding_dim,args.windows_size)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.95)
    loss_fn = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    training_data = WordData()
    train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)
    for epoch in range(args.NIterators):
        train_loss = 0
        model.train()
        for feature, label in train_dataloader:
            model = model.to(device)
            feature = feature.to(device)
            label = label.to(device)
        
            y_train_pred = model(feature)
        
            loss = loss_fn(y_train_pred, label)
            train_loss = train_loss + loss
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_dataloader)
        print(f"Epoch:{epoch} | Training Loss : {train_loss}")  
        
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    
if __name__=='__main__':
    main()
