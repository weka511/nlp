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
    Allow a program to be structured as a set of Command objects, each
    of which registers a commens string to be selected by the user.
'''

__version__ = '1.0'
__author__ = 'Simon Crase'

from abc import ABC, abstractmethod
from pathlib import Path
from time import time
import numpy as np
from shared.utils import Logger, get_seed

class Command(ABC):
    '''
    This class is the parent for all the tasks performed by this program.
    It provides a list of ecceptable commands (used by parse_args(),
    and its subclasses are each responsible for one task.
    '''
    choices = {}

    @staticmethod
    def append(commands):
        '''
        Prepare one of more commands
        
        Parameters:
            cpmmands   List of Commands that are available to user
        '''
        for command in commands:
            Command.choices[command.key] = command

    @staticmethod
    def get_choices():
        '''
        Get list of available Commands
        '''
        return [key for key in Command.choices.keys()]

    @staticmethod
    def get_command(key):
        '''
        Get Command given the user's specification
        
        Parameters:
            key      Command name specified by user
        '''
        return Command.choices[key]

    def __init__(self, key):
        '''
        Used when command is created to set the name that the user specifies
        
        Parameters:
            key      The name that the user specifies for this command
        '''
        self.key = key

    def execute(self, args):
        '''
        Set up parameters needed by Command, then execute it
        
        Parameters:
            args    Command line parameters as parsed by parse_args()
        '''
        with Logger(Path(__file__).stem, path=args.logs) as _:
            start = time()
            for key, value in vars(args).items():
                Logger.get_instance().log(f'{__file__} {Logger.get_line()} {key} = {value}')

            seed = get_seed(args.seed,
                            notify=lambda s: Logger.get_instance().log(f'{__file__} {Logger.get_line()}'
                                                                       f' Created new seed {s}'))
            
            self.seed_hook(seed)
            
            self._execute(args, rng=np.random.default_rng(seed))

            elapsed = time() - start
            minutes = int(elapsed / 60)
            seconds = elapsed - 60 * minutes
            Logger.get_instance().log(f'{__file__} {Logger.get_line()} Elapsed Time {minutes} m {seconds:.2f} s')         

    @abstractmethod
    def _execute(self, args, rng=np.random.default_rng()):
        '''
        Perform command
        
        Parameters:
            args       Command line parameters as parsed by parse_args()
            rng        Random number generator
        '''
        
    def seed_hook(self,seed):
        '''
        Allow subclass to use changed seed (e.g. pass to pytorch)
        '''
        pass
    