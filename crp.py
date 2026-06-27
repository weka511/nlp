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

'''
    This module clusters data using a Distance Dependent Chinese Restaurant Process,
    David M. Blei, Peter I. Frazier; 
    Journal of Machine Learning Research 12(74):2461−2488, 2011,
    https://www.jmlr.org/papers/v12/blei11a.html
'''

from abc import ABC, abstractmethod
from unittest import TestCase, main, skip
import numpy as np
from shared.utils import Logger

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
        links = ','.join([f'{a}->{b}' for a, b in self.links.items()])
        return f'Table {self.seq} [{len(self)}]: {links}'

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
            raise  ValueError('Node must link to itself in an otherwise empty table, or to a distinct node')

    def break_link(self,node):
        del self.links[node]
        
    #def relink(self,node,end):
        #self.links[node] = end
    
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
    
    def split(self,nodes,current_table, target_table):
        for node in nodes:
            break
        #print(current_table, target_table)
        #print(nodes)
        #z=0
        #pass
    
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
        m        Number of points to be classified (from one dimension of distance matrix)
        fd       Array of distances between nodes, after applying decay function
        rng      Random number generator
        alpha    The scaling parametere from Blei and Fraxier, equation (2)
        links    FIXME
        tables   Lookup table to find which Table holds each node
        logger   For logging messages
    '''

    UNASSIGNED = -1

    def __init__(self, distances, f=NoDecay(), rng=np.random.default_rng(), alpha=2.0, logger=None):
        '''
        Parameters:
            distances  Array of distances between nodes, after applying decay function
            f          Decay function for distances
            rng        Random number generator
            alpha      The scaling parametere from Blei and Fraxier, equation (2)
            logger     For logging messages
        '''
        self.rng = rng
        self.m, n = distances.shape
        assert self.m == n
        self.fd = f(distances)
        self.rng = rng
        self.alpha = alpha
        #self.links = np.full((self.m), ChineseRestaurantProcess.UNASSIGNED, dtype=int)
        self.tables = np.empty((self.m), dtype=Table)  # FIXME
        self.logger = logger

    def build(self):
        '''
        Allocate each node to a table. It randomizes the order of nodes, then adds
        one at a time. If the 
        '''
        indices = self.rng.permutation(self.m)
        for i in range(self.m):
            this_node = int(indices[i])
            link_to = int(self.rng.choice(indices[:i + 1], p=self._get_p(i, initial=True)))
            if this_node == link_to:
                self._link(this_node,link_to,table=Table())
            else:
                self._link(this_node,link_to,table=self.tables[link_to])
        
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
        for node_being_considered in range(self.m):
            current_table = self.tables[node_being_considered]
            link_to = int(self.rng.choice(self.m, p=self._get_p(node_being_considered)))
            if node_being_considered == link_to: continue # Linking to same node, so no change

            target_table = self.tables[link_to]
            if current_table == target_table:  # Link to different node in same table
                self.relink(current_table,node_being_considered,target_table,link_to)
 
            else:    # Link to a node in another Table
                self._link_to_separate_table(current_table,node_being_considered,target_table)
                
    def relink(self,current_table,node_being_considered,target_table,link_to):
        '''
        Link to different node in same table
        '''
        
        # Find out whether breaking the olf link will split the table in two
        
        components = self.get_components_once_broken(current_table,node_being_considered)
        
        match (len(components)):
            case 1:  # Table will not be split
                self.logger.log('One component')
                current_table.break_link(node_being_considered)
                current_table[node_being_considered] = link_to
                
            case 2:  # Table might be splt
                self.logger.log('Two components')
                if node_being_considered in components[0] and link_to in components[0]:
                    current_table.split(components[0],node_being_considered, link_to)
                elif node_being_considered in components[1] and link_to in components[1]:
                    current_table.split(components[1],node_being_considered, link_to)
                else: # New link will restore things to a single table
                    self.logger.log('New link will restore things to a single table')
                    current_table.break_link(node_being_considered)
                    current_table[node_being_considered] = link_to

            case _:
                self.logger.log('WTF',level=Logger.ERROR)    
                
    def get_components_once_broken(self,table,node):
        '''
        Find out whether deleting one link will split cluster into two parts
        
        Parameters:
            table
            node
            
        Returns:
           The components of the graph on the assumption thsat the link from node has been deleted
        '''
        return Table.create_cc(table.create_edge_list(omit=node))
        
    def _link_to_separate_table(self,current_table,node_being_considered,target_table):
        edge_list = current_table.create_edge_list(omit=node_being_considered)
        cc = Table.create_cc(edge_list)
        match len(cc):
            case 1:
                #self.logger.log(f'Moving {cc[0]} to {target_table}')
                #target_table.join(cc[0], current_table)
                #for node in cc[0]:
                    #self.tables[node] = target_table
                #current_table.clear()
                self.logger.log(target_table)
                self.logger.log(current_table)
            case 2:
                self.logger.log(f'{cc}')
                #if node_being_considered in cc[0]:
                    #target_table.join(cc[0], current_table)
                    #for node in cc[0]:
                        #self.tables[node] = target_table
                    #current_table.delete(cc[0])
                #elif node_being_considered in cc[1]:
                    #target_table.join(cc[1], current_table)
                    #for node in cc[1]:
                        #self.tables[node] = target_table
                    #current_table.delete(cc[1])
                #else:
                    #self.logger.log('WTF')
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

    def _get_p(self, node, initial=False):
        '''
        Calculate probabilities for assignments after Blei & Frazier equation (2)
        
        Parameters:
            node    The node we are currently considering
            initial    If we are ineitializing we want to produce 
                       a vector that is shorter than self.m
        '''
        n = node + 1 if initial else self.m
        p = np.empty((n))
        for i in range(n):
            p[i] = 1 / self.fd[node, i] if i != node else self.alpha
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

    class TestTableUnlink(TestCase):
    
        def test_table_create(self):
            t0 = Table()
            t0[1] = 1
            t0[2] = 1
            t0[3] = 2
            t0[4] = 3
            t0[5] = 3
            t0[6] = 5
            t0[7] = 5
            t0[8] = 7
            t0.relink(5,4)
            self.assertEqual (t0[5], 4)
            
    main()
