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
from sys import float_info
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

    def __init__(self, distances,
                 f=NoDecay(), rng=np.random.default_rng(),
                 alpha=2.0, logger=None):
        self.rng = rng
        self.m, self.n = distances.shape
        self.distances = f(distances)
        assert self.m == self.n
        self.rng = rng
        self.alpha = alpha
        self.links = np.full((self.m), ChineseRestaurantProcess.UNASSIGNED, dtype=int)
        self.tables = np.empty((self.m), dtype=Table)
        self.logger = logger
        
    def build(self):
        '''
        Allocate each element to a table
        '''
        indices = self.rng.permutation(self.m)
        for i in range(self.m):
            current = int(indices[i])
            link_to = int(self.rng.choice(indices[:i + 1], p=self._get_p_init(i)))
            self._link(current,link_to,
                      table = Table() if current == link_to else self.tables[link_to])
   
        for table in Table.tables:
            self.logger.log(table)
            
    def _link(self,start,end, table=None):
        '''
        Link two nodes, and also record that the first node is in the table
        
        Parameters:
            start    This node is to be linked to other node
            end      This is the node to link to
            table    The link lives in this table
        '''
        table.link(start, end)
        self.tables[start] = table
        assert self.tables[end] == table
        
    def _get_p_init(self, seq):
        '''
        Calculate probabilities for initialization
        
        Parameters:
            seq    Index of element that is being added     
        '''
        p = 1 / (self.distances[seq, :] + float_info.min)
        p[seq] = self.alpha
        p = p[:seq + 1]
        return p / p.sum()    
    
class TestTable(TestCase):
    def test1(self):
        self.assertEqual(1,1)
        
if __name__ == '__main__':
    main()
    