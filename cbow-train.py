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

'''Train Continuous Bag of Words'''

from argparse import ArgumentParser
from csv import reader
from pathlib import Path
from queue import Queue
from time import time
from threading import Thread

import numpy as np

from shared.utils import Logger, user_has_requested_stop, get_seed
from vocabulary import Vocabulary
from cbow2 import Model, OneHotFactory, GradientDescent, CrossEntropyLoss

__version__ = '1.0'
__author__ = 'Simon Crase'

class DataLoader:
    '''
    This class reads stored training examples. It is meant to run in separate thread from training.
    '''
    Sentinel = -1 # Used to inform tharing thread that we have no more data for it
    
    def __init__(self,maxsize=8,data='data',examples='examples'):
        self.maxsize = maxsize
        self.pipeline = Queue(maxsize=maxsize)
        self.root_dir = Path(data) / examples
        self.vocabulary = Vocabulary.create((self.root_dir / 'vocabulary').with_suffix('.pkl'))
        
    def __len__(self):
        '''
        Returns numbert of token in vocabulary
        '''
        return len(self.vocabulary)

    def load(self,worker):
        '''
        Read data from saved examples and queue it, so
        it can be read by worker thread.
        
        Parameters:
            worker     The worker thread that extract data from queue
        '''
        with open((self.root_dir / 'progress').with_suffix('.txt'),'a') as progress:
            running = True
            for path in self.root_dir.rglob("*.csv"):
                if not running: break
                if not path.is_file(): continue
 
                Logger.get_instance().log(f'{__file__} {Logger.get_line()} File: {path}')      
                with open(path,newline='') as in_file:
                    for row in reader(in_file,delimiter=','):
                        tokens = [int(w) for w in row]
                        self.pipeline.put(tokens)
                        
                progress.write(f'{path.stem}\n')
                
                if user_has_requested_stop():
                    running = False
                    Logger.get_instance().log(f'{__file__} {Logger.get_line()} Stopping within {self.maxsize} steps')
                    break
   
        self.pipeline.put([DataLoader.Sentinel])   
        Logger.get_instance().log(f'{__file__} {Logger.get_line()}')
        worker.join()
        Logger.get_instance().log(f'{__file__} {Logger.get_line()}')    
        
    def consume(self):
        '''
        This is called from the worker thread to extract data from queue
        '''
        while True:
            tokens = self.pipeline.get()
            if tokens[0] == DataLoader.Sentinel: return
            mid_point = len(tokens) // 2
            yield tokens[:mid_point] + tokens[mid_point+1:], tokens[mid_point]        
            self.pipeline.task_done()
        
def parse_args():
    data = './data'
    logs = './logs'
    examples = 'examples'
    n = 300
    N = 50
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generation')
    parser.add_argument('--data', default=data, help=f'Path to data files [{data}]')
    parser.add_argument('--logs', default=logs, help=f'Location for storing log files [{logs}]')
    parser.add_argument('-n', type=int, default=n)
    parser.add_argument('-N', type=int, default=N)
    parser.add_argument('--examples', default=examples, help=f'Path to examples files [{examples}]')
    return parser.parse_args()

def train(dataloader,encoder,model,loss_fn,optimizer):
    '''
    Train for one datum
    
    Parameters:
        dataloader   Reads examples and formats as feature and label
        encoder      Convert tokens to word vectors
        model        The model being trained
        loss_fn      Computer training loss
        optimizer    Use the adjust weights to minimize loss
    '''
    for feature,label in dataloader.consume():
        label = encoder.create(label)
        prediction = model(feature)
        loss = loss_fn(prediction,label)
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Loss={loss}')
        loss_fn.backward()
        optimizer.step()

def main():
    '''
    Train Continuous Bag of Words'
    '''
    args = parse_args()
    with Logger(Path(__file__).stem, path=args.logs) as _:
        start = time()
        for key, value in vars(args).items():
            Logger.get_instance().log(f'{__file__} {Logger.get_line()} {key} = {value}')

        seed = get_seed(args.seed,
                        notify=lambda s: Logger.get_instance().log(f'{__file__} {Logger.get_line()}'
                                                                   f' Created new seed {s}'))
        
        dataloader = DataLoader(examples=args.examples)
        encoder = OneHotFactory(n=len(dataloader))
        model = Model(m=len(dataloader), n=args.n,encoder=encoder,rng=np.random.default_rng(seed=seed))
        loss_fn = CrossEntropyLoss(model)
        optimizer = GradientDescent(model,loss_fn,lr=0.01)      
        worker = Thread(target=train, args=[dataloader,encoder,model,loss_fn,optimizer],daemon=True)
        worker.start()
        dataloader.load(worker)
 
    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')

if __name__ == '__main__':
    main()
