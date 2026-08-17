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

from argparse import ArgumentParser
from csv import reader
from pathlib import Path
from queue import Queue
from time import time
from threading import Thread

import numpy as np
from shared.utils import Logger, user_has_requested_stop, get_seed

from cbow2 import Model, OneHotFactory, GradientDescent, CrossEntropyLoss
class DataLoader:
    def __init__(self,maxsize=2):
        self.pipeline = Queue(maxsize=2)
        self.there_is_data = True
        
    def load(self):
        with open(r'C:\Users\weka5\nlp\data\examples\A\A0\A00.csv',newline='') as in_file:
            for row in reader(in_file,delimiter=','):
                tokens = [int(w) for w in row]
                self.pipeline.put(tokens)
                
    def consume(self):
        while self.there_is_data:
            tokens = self.pipeline.get()
            mid_point = len(tokens) // 2
            yield tokens[:mid_point] + tokens[mid_point+1:], tokens[mid_point]        
            self.pipeline.task_done()
        
    def join(self):
        Logger.get_instance().log(f'{__file__} {Logger.get_line()}')
        self.pipeline.join()
        Logger.get_instance().log(f'{__file__} {Logger.get_line()}')
        
def parse_args():
    data = './data'
    logs = './logs'
    m = 31
    n = 300
    N = 50
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generation')
    parser.add_argument('--data', default=data, help=f'Path to data files [{data}]')
    parser.add_argument('--logs', default=logs, help=f'Location for storing log files [{logs}]')
    parser.add_argument('-m', type=int, default=m)
    parser.add_argument('-n', type=int, default=n)
    parser.add_argument('-N', type=int, default=N)
    return parser.parse_args()

def worker(dataloader,encoder,model,loss_fn,optimizer):
    for feature,label in dataloader.consume():
        label = encoder.create(label)
        prediction = model(feature)
        loss = loss_fn(prediction,label)
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Loss={loss}')
        loss_fn.backward()
        optimizer.step()

def main():
    args = parse_args()
    with Logger(Path(__file__).stem, path=args.logs) as _:
        start = time()
        for key, value in vars(args).items():
            Logger.get_instance().log(f'{__file__} {Logger.get_line()} {key} = {value}')

        seed = get_seed(args.seed,
                        notify=lambda s: Logger.get_instance().log(f'{__file__} {Logger.get_line()}'
                                                                   f' Created new seed {s}'))
        rng = np.random.default_rng(seed=seed)
        encoder = OneHotFactory(n=args.m)
        model = Model(m=args.m,
                      n=args.n,
                      encoder=encoder,
                      rng=rng)

        loss_fn = CrossEntropyLoss(model)

        optimizer = GradientDescent(model,loss_fn,lr=0.01)
        
        dataloader = DataLoader()
        Thread(target=worker, args=[dataloader,encoder,model,loss_fn,optimizer],daemon=True).start()
        dataloader.load()
        dataloader.join()
        
 
    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')


if __name__ == '__main__':
    main()
