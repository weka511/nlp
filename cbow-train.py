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
from pathlib import Path
from time import time

import numpy as np
from shared.utils import Logger, user_has_requested_stop, get_seed

from cbow2 import Model, OneHotFactory, GradientDescent, CrossEntropyLoss

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
        for i in range(args.N):
            feature = [0, 2, 4, 5]
            label = encoder.create(3)
            prediction = model(feature)
            loss = loss_fn(prediction,label)
            print (loss)
            loss_fn.backward()
            optimizer.step()

    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')


if __name__ == '__main__':
    main()
