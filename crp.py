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

from abc import ABC, abstractmethod
from unittest import TestCase, main, skip
import numpy as np


class Table:
    '''
    The Chinese Restaurant Process allocates elements to a Table. Each element is
    a number, e.g. the index of a word in a vocabulary, so all structure information 
    has to be stored in the Table.
    
    Attributes:
        links  A list of links [...(from,to), ....]
        seq    A  unique id for this Table
    '''
    current_seq = -1   # Used to assign unique sequnce numbers
    tables = []        # List of all tables

    @staticmethod
    def create_seq():
        Table.current_seq += 1
        return Table.current_seq
    
    def __init__(self):
        '''
        Initialize links to an empty list, allocate a sequence number to the table,
        and store newly created table in a list of table.
        '''
        self.links = {}
        self.seq = Table.create_seq()
        Table.tables.append(self)

    def __eq__(self, table):
        '''
        Two tables are deemed equal if they have the same sequence number
        '''
        return self.seq == table.seq

    def __len__(self):
        '''
        The length of the table is the number of links
        '''
        return len(self.links)

    def __str__(self):
        '''
        Format all the links in the table for display
        '''
        s = ','.join([f'{a}->{b}' for a, b in self.links.items()])
        return f'Table {self.seq} [{len(self)}]: {s}'

    def __getitem__(self, key):
        '''
        Get the node that the supplied node links to
        
        Parameters:
            key        Stat of link
        '''
        return self.links[key]

    def __setitem__(self, start, end):
        '''
        Link two elements together
        
        Parameters:
            start      Link will be from this node
            end        Link to this node
        '''
        if start in self.links: raise ValueError(f'{start} is already in links.')
        
        if (len(self) == 0) == (start == end):
            self.links[start] = end
        else:
            raise  ValueError('Node must link to itself in an otherwise emepty table, or to a distinct node')

    
    def join(self, nodes, current_table):
        for node in nodes:
            self[node] = current_table[node]

    def clear(self):
        self.links = {}

    def delete(self, nodes):
        for node in nodes:
            del self.links[node]

    def create_edge_list(self, omit=-1):
        '''
        Convert links to edge-link format so we can use create_cc, 
        optionally omitting one specified link
        
        Parameters:
             omit      Link to be omitted
        '''
        return [(start, end) for start, end in self.links.items() if start != omit and start != end]

    def generate_links(self):
        '''
        Used to iterate through all nodes at Table
        '''
        for start, end in self.links.items():
            yield start, end

    @staticmethod
    def create_cc(g):
        '''
        Create a list of connected components for a graph
        '''
        def create_vertices(g):
            product = set()
            for a, b in g:
                product.add(a)
                product.add(b)
            return product

        def create_augmented(g):
            product = {}
            for a, b in g:
                if a not in product:
                    product[a] = []
                if b not in product:
                    product[b] = []
                product[a].append(b)
                product[b].append(a)
            return product

        def dfs(vertex, g_augmented, visited):
            component = set()
            visited.append(vertex)
            component.add(vertex)
            for child in g_augmented[vertex]:
                if child not in visited:
                    component = component.union(dfs(child, g_augmented, visited))
            return component

        g_augmented = create_augmented(g)
        visited = []
        return [dfs(vertex, g_augmented, visited)
                for vertex in create_vertices(g)
                if vertex not in visited]


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
    
    Attributes:
        m
        distances
        rng
        alpha
        links
        tables
        logger
    '''

    UNASSIGNED = -1

    def __init__(self, distances, f=NoDecay(), rng=np.random.default_rng(), alpha=2.0, logger=None):
        '''
        Parameters:
            distances
            f
            rng
            alpha
            logger
        '''
        self.rng = rng
        self.m, n = distances.shape
        assert self.m == n
        self.distances = f(distances)
        self.rng = rng
        self.alpha = alpha
        self.links = np.full((self.m), ChineseRestaurantProcess.UNASSIGNED, dtype=int)
        self.tables = np.empty((self.m), dtype=Table)
        self.logger = logger

    def build(self):
        '''
        Allocate each node to a table. It randomizes the order of nodes, then adds
        one at a time. If the 
        '''
        indices = self.rng.permutation(self.m)
        for i in range(self.m):
            current_node = int(indices[i])
            link_to = int(self.rng.choice(indices[:i + 1], p=self._get_p(i, initial=True)))
            table = None
            if current_node == link_to:
                table = Table()
            else:
                table = self.tables[link_to]
            self._link(current_node, link_to, table)
        
        assert self.m == sum(len(table) for table in Table.tables)

    def generate_tables(self):
        '''
        Used to iterate through all tables
        '''
        for table in Table.tables:
            yield table

    def gibbs(self):
        '''
        Perform one gibbs step - WIP
        '''
        for current in range(self.m):
            current_table = self.tables[current]
            link_to = int(self.rng.choice(self.m, p=self._get_p(current)))
            if current == link_to:
                continue # Linking to same node, so no change

            target_table = self.tables[link_to]
            if current_table == target_table:  # Link to different node in same table
                current_table[current] = link_to
            else:    # Link to a node in another Table
                edge_list = current_table.create_edge_list(omit=current)
                cc = Table.create_cc(edge_list)
                match len(cc):
                    case 1:
                        self.logger.log(f'Moving {cc[0]} to {target_table}')
                        target_table.join(cc[0], current_table)
                        for node in cc[0]:
                            self.tables[node] = target_table
                        current_table.clear()
                        self.logger.log(target_table)
                        self.logger.log(current_table)
                    case 2:
                        self.logger.log(f'{cc}')
                        if current in cc[0]:
                            target_table.join(cc[0], current_table)
                            for node in cc[0]:
                                self.tables[node] = target_table
                            current_table.delete(cc[0])
                        elif current in cc[1]:
                            target_table.join(cc[1], current_table)
                            for node in cc[1]:
                                self.tables[node] = target_table
                            current_table.delete(cc[1])
                        else:
                            self.logger.log('WTF')
                    case _:
                        self.logger.log(f'oops: len(cc)={len(cc)}')

    def _link(self, start, end, table=None):
        '''
        Link two nodes, and also record that the first node is in the table
        
        Parameters:
            start    This node is to be linked to other node
            end      This is the node to link to
            table    The link lives in this table
        '''
        table[start] = end
        self.tables[start] = table
        assert self.tables[end] == table

    def _get_p(self, current, initial=False):
        '''
        Calculate probabilities for assignments after Blei & Frazier equation (2)
        
        Parameters:
            current    The node we are currently considering
            initial    If we are ineitializing we want to produce 
                       a vector that is shorter than self.m
        '''
        n = current + 1 if initial else self.m
        p = np.empty((n))
        for i in range(n):
            p[i] = 1 / self.distances[current, i] if i != current else self.alpha
        return p / p.sum()


if __name__ == '__main__':
    class TestTable(TestCase):
    
        def test_table_create(self):
            t0 = Table()
            self.assertEqual(0, t0.seq)
            t1 = Table()
            self.assertEqual(1, t1.seq)  
            
        def test_table_link(self):
            t0 = Table()
            self.assertEqual(0,len(t0))
            with self.assertRaises(ValueError):
                t0[5] = 7
            self.assertEqual(0,len(t0))
            t0[5] = 5
            t0[7] = 5
            self.assertEqual(2,len(t0))
            t0[1] = 5
            self.assertEqual(3,len(t0))  
            t0[2] = 7
            self.assertEqual(4,len(t0))
            with self.assertRaises(ValueError):
                t0[7] = 9
            
        def test_cc(self):
            '''
            This is the test case from https://rosalind.info/problems/cc/
            '''
            cc = Table.create_cc([
                (1, 2),
                (1, 5),
                (5, 9),
                (5, 10),
                (9, 10),
                (3, 4),
                (3, 7),
                (3, 8),
                (4, 8),
                (7, 11),
                (8, 11),
                (11, 12),
                (8, 12),
                (6, 6)
            ])
            self.assertEqual(3, len(cc))
            self.assertCountEqual([1, 2, 5, 9, 10], cc[0])
            self.assertCountEqual([3, 4, 7, 8, 11, 12], cc[1])
            self.assertCountEqual([6], cc[2])
        
    main()
