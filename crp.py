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
#  along with this program.   If not, see <https://www.gnu.org/licenses/>.

from abc import ABC,abstractmethod
from unittest import TestCase, main,skip
import numpy as np

class Table:
    '''
    The Chinese Restaurant Process allocates elements to a Table.
    
    Attributes:
        links
        seq
    '''
    seq = -1
    tables = []

    def __init__(self):
        self.links = {}
        Table.seq += 1
        self.seq = Table.seq
        Table.tables.append(self)

    def __eq__(self,table):
        '''
        Two tables are deemed equal if they have the same sequence number
        '''
        return self.seq == table.seq
    
    def __len__(self):
        return len(self.links)

    def __str__(self):
        s = ','.join([f'{a}->{b}' for a,b in self.links.items()])
        return f'Table {self.seq}: {s}'
    
    def __getitem__(self,key):
        return self.links[key]

    def link(self, start, end):
        '''
        Link two elements together
        
        Parameters:
            start
            end
        '''
        self.links[start] = end
        
    def join(self,nodes,current_table):
        for node in nodes:
            self.link(node,current_table[node])
    
    def clear(self):
        self.links = {}
    
    def delete(self,nodes):
        for node in nodes:
            del self.links[node]
        
    def create_edge_list(self,split=-1):
        '''
        Convert links to edge-link format so we can use create_cc, 
        optionally omitting one specified link
        '''
        return [(start,end) for start,end in self.links.items() if start != split and start != end]

    def print_links(self):
        for start, end in self.links.items():
            print(f'....{start}->{end}')


class DecayFunction(ABC):
    '''
    This class is used to model the decay function of Blei and Frazier
    '''
    @abstractmethod
    def __call__(self, distance):
        '''
        Apply decay function
        '''


class NoDecay(DecayFunction):
    '''
    Pass distance through unchanged
    '''

    def __call__(self, distance):
        return distance
        
class ChineseRestaurantProcess:
    '''
    This class represents a distance dependent Chinese Restaurant Process. 
    '''
    
    UNASSIGNED = -1

    def __init__(self, mutual_information,
                 f=NoDecay(), rng=np.random.default_rng(),
                 alpha=2.0, logger=None):
        #self.fd = f(1 / mutual_information)
        self.rng = rng
        self.m, self.n = mutual_information.shape
        assert self.m == self.n
        self.rng = rng
        self.alpha = alpha
        self.links = np.full((self.m), ChineseRestaurantProcess.UNASSIGNED, dtype=int)
        self.tables = np.empty((self.m), dtype=Table)
        self.logger = logger    
    
class TestTable(TestCase):
    def test1(self):
        self.assertEqual(1,1)
        
if __name__ == '__main__':
    main()
    