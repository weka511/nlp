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
    tables = []        # List of all tables

    @staticmethod
    def reset():
        Table.tables = [] 
    

    def __init__(self):
        '''
        Initialize links to an empty list, allocate a sequence number to the table,
        and store newly created table in a list of table.
        '''
        self.links = {}
        self.seq = len(Table.tables)
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
        if start in self.links:
            raise ValueError(f'{start} is already in links.')

        if (len(self) == 0) == (start == end):
            self.links[start] = end
        else:
            raise ValueError('Node must link to itself in an otherwise empty table, or to a distinct node')
        
    def break_at(self,node):
        components = Table.create_connected_components(self.create_edge_list(omit=node))
        match len(components):
            case 1:
                pass
            case 2:
                if not node in components[1]:
                    components = [components[1],components[0]]
                table1 = Table()
                table1[node] = node
                to_delete = []
                for a,b in self.links.items():
                    if a in components[0] and b in components[0]: 
                        pass # keep link
                    elif a in components[1] and b in components[1]:
                        table1[a] = b
                        to_delete.append(a)
                    else:
                        to_delete.append(a)
                for a in to_delete:
                    del self.links[a]
                return table1
            case _:
                raise RuntimeError(f'There are {len(cc)} connected components')
            
    def join(self,node,table_split,node_to_connect):
        self.links |= table_split.links
        self.links[node_to_connect] = node
        table_split.links.clear()

    def break_link(self, node):
        try:
            del self.links[node]
        except KeyError:
            for key, value in self.links.items():
                print(key, value)
 

    #def join(self, nodes, current_table):
        #for node in nodes:
            #self[node] = current_table[node]

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

    def split(self, nodes):
        '''
        Split a bunch of nodes off into a separate table
        
        Parameters:
            nodes      List of nodes to be removed
        '''
        new_table = Table()
        for node in nodes:
            destination = self[node]
            new_table.links[node] = destination
            self.break_link(node)

        # TODO: need at least one cycle

    def get_nodes(self):
        nodes = set()
        for start, end in self.links.items():
            nodes.add(start)
            nodes.add(end)
        return list(nodes)

    def verify_consistency(self, fix=False):
        '''
        Verify that evey node has an exit
        '''
        ins = set()
        outs = set()
        for start, end in self.links.items():
            ins.add(end)
            outs.add(start)
        gap = ins - outs
        if len(gap) == 0:
            return True
        elif len(gap) == 1:
            singleton = gap.pop()
            self.links[singleton] = singleton
        else:
            print(gap)
            return False

    @staticmethod
    def create_connected_components(g):
        '''
        Create a list of connected components for a graph
        
        Parameters:
            g       Graph in edge list format
        '''
        def create_vertices(g):
            '''
            Construct the set up all vertices in the graph
            '''
            product = set()
            for a, b in g:
                product.add(a)
                product.add(b)
            return product

        def create_augmented(g):
            '''
            Create a new graph that has forward and backward links.
            
            Parameters:
                g       Graph in edge list format
                
            Returns:
               Newly constructed graph
            '''
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
            '''
            This is the heart of the algorithm. It performs Depth First Search 
            to visit every node in a component once and only once.
            
            Parameters:
                vertex         A node that has not been visted yet
                g_augmented    Graph, augmented with forward and backward links
                visited        List of nodes that have already been visited
                
            Returns:
                Newly constucted component
            '''
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


class Chooser(ABC):
    '''
    This class is the parent of classes that choose which node to link to.
    It allows the real Chooser to be mocked for testing
    '''
    @abstractmethod
    def choose(self, n, initial=False):
        '''
        Choose which node to link to
        '''


class DistanceDependentChooser(Chooser):
    '''
    This class forms part of a distance dependent Chinese Restaurant Process. It chooses which node to link to
    
    Attributes:
        fd
        rng
        alpha
        m
    '''

    def __init__(self, d, f=NoDecay(), rng=np.random.default_rng(), alpha=2.0, m=None):
        '''
            f          Decay function         
            rng        Random number generator
            alpha      The scaling parametere from Blei and Frazier, equation (2)
        '''
        self.fd = f(d)
        self.rng = rng
        self.alpha = alpha
        self.m = m

    def get_indices(self, m):
        return self.rng.permutation(m)

    def choose(self, n, initial=False):
        '''
        Choose which node to link to
        '''
        return self.rng.choice(n + 1 if initial else self.m, p=self._get_p(n, initial=initial))

    def _get_p(self, node, initial=False):
        '''
        Calculate probabilities for assignments after Blei & Frazier equation (2)
        
        Parameters:
            node    The node we are currently considering
            initial If we are initializing we want to produce 
                     a vector that is shorter than self.m
        '''
        n = node + 1 if initial else self.m
        p = np.empty((n))
        for i in range(n):
            p[i] = 1 / self.fd[node, i] if i != node else self.alpha
        return p / p.sum()


class ChineseRestaurantProcess:
    '''
    This class represents a Chinese Restaurant Process. 
    
    Attributes:
        tables   Lookup table to find which Table holds each node
        logger   For logging messages
    '''

    UNASSIGNED = -1

    def __init__(self, chooser=None, logger=None, m=None):
        '''
        Parameters:
            tables
            m
            logger     For logging messages
        '''
        self.m = m
        self.tables = np.empty((self.m), dtype=Table)
        self.logger = logger
        self.chooser = chooser

    def build(self):
        '''
        Allocate each node to a table. It randomizes the order of nodes, then adds
        one at a time. If the 
        '''
        for this_node in range(self.m):
            link_to = self.chooser.choose(this_node, initial=True)
            if this_node == link_to:
                self._link(this_node, link_to, table=Table())
            else:
                self._link(this_node, link_to, table=self.tables[link_to])

        #print (self.m)
        #print ([len(table) for table in Table.tables])
        assert self.m == sum(len(table) for table in Table.tables)

        for table in Table.tables:
            table.verify_consistency()

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
        for node in range(self.m):
            table = Table.tables[node]
            link_to = self.chooser.choose(node)
            if node == link_to:
                continue # Linking to same node, so no change

            target_table = Target.tables[link_to]
            if table == target_table:  # Link to different node in same table
                self._move_link(node, table, link_to)
                if not table.verify_consistency(fix=True):
                    print(table)
                    z = 0

            else:    # Link to a node in another Table  FIXME
                pass#self._link_to_separate_table(table,node,target_table)

   
    
    def _move_link(self, node, table, link_to):
        '''
        Link to different node in same table
        
        Parameters:
            node     The node whose link is to be changed
            table    The table that the node belongs to
            link_to  The node we will link to
        '''

        # Find out whether breaking the old link will split the table in two

        components = self._get_components_once_link_broken(node, table)

        match (len(components)):
            case 1:  # Table will not be split by breaking link
                table.break_link(node)
                table[node] = link_to

            case 2:  # Table might be split by breaking link
                if node in components[0] and link_to in components[0]:
                    table.split(components[1])
                elif node in components[1] and link_to in components[1]:
                    table.split(components[0])
                else: # New link will restore things to a single table, so no split after all
                    pass

                table.break_link(node)
                table[node] = link_to

            case _:
                self.logger.log(f'Length of components is {len(components)}, should be 1 or 2', level=Logger.ERROR)

    def _get_components_once_link_broken(self, node, table):
        '''
        Find out whether deleting one link will split cluster into two parts
        
        Parameters:
            node     The node whose link we are about to break
            table    The table in which the node lives
  
        Returns:
           The components of the graph on the assumption thsat the link from node has been deleted
        '''
        return Table.create_connected_components(table.create_edge_list(omit=node))

    def _link_to_separate_table(self, current_table, node_being_considered, target_table):
        components = self._get_components_once_link_broken(current_table, node_being_considered)
        match len(components):
            case 1:
                #self.logger.log(f'Moving {components[0]} to {target_table}')
                #target_table.join(components[0], current_table)
                #for node in components[0]:
                    #self.tables[node] = target_table
                #current_table.clear()
                self.logger.log(target_table)
                self.logger.log(current_table)
            case 2:
                self.logger.log(f'{components}')
                #if node_being_considered in components[0]:
                    #target_table.join(components[0], current_table)
                    #for node in components[0]:
                        #self.tables[node] = target_table
                    #current_table.delete(components[0])
                #elif node_being_considered in components[1]:
                    #target_table.join(components[1], current_table)
                    #for node in components[1]:
                        #self.tables[node] = target_table
                    #current_table.delete(components[1])
                #else:
                    #self.logger.log('WTF')
            case _:
                self.logger.log(f'oops: len(components)={len(components)}')

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

if __name__ == '__main__':
    class MockChooser(Chooser):
        '''
        This class allows tests to build Tables
        '''
        def __init__(self,nodes):
            self.nodes = nodes
            self.pos = -1
            
        def choose(self, n, initial=False):
            self.pos += 1
            return self.nodes[self.pos]
        
    class TestFigure3(TestCase):
        '''
        Tests based on Figure 3 of Blei & Frazier
        '''        
        def setUp(self):
            chooser = MockChooser([0,0,1,2,4,4])
            self.crp = ChineseRestaurantProcess(chooser=chooser,m=6)
            self.crp.build()            
        
        def test_build_tables(self):
            '''
            This test verifies that we can construct the graph in the first panel of Figure 3 of Blei & Frazier
            '''
            self.assertEqual(2,len(Table.tables))
            table0 = Table.tables[0]
            table1 = Table.tables[1]
            self.assertNotEqual(table0,table1)
            self.assertEqual(0,table0[0])
            self.assertEqual(0,table0[1])
            self.assertEqual(1,table0[2])
            self.assertEqual(2,table0[3])
            self.assertEqual(4,table1[4])
            self.assertEqual(4,table1[5])
            
        def test_break_link(self):
            '''
            This test verifies that we can construct the graph in lines 2 and 3 Figure 3 of Blei & Frazier
            '''            
            table = Table.tables[0]
            table_split = table.break_at(2)
            self.assertEqual(3,len(Table.tables))
            table_join_to = self.crp.tables[4]
            table_join_to.join(4,table_split,2)
            self.assertEqual(6,sum([len(t) for t in Table.tables]))
            self.assertEqual(4,len(table_join_to))
            self.assertEqual(0,Table.tables[0][0])
            self.assertEqual(0,Table.tables[0][1])
            self.assertEqual(4,Table.tables[1][4])
            self.assertEqual(4,Table.tables[1][5])
            self.assertEqual(4,Table.tables[1][2])
            self.assertEqual(2,Table.tables[1][3])
        
        def tearDown(self):
            Table.reset()
            
    class TestTable(TestCase):

        def test_table_create(self):
            t0 = Table()
            self.assertEqual(0, t0.seq)
            t1 = Table()
            self.assertEqual(1, t1.seq)
            self.assertNotEqual(t0,t1)
            self.assertEqual(2,len(Table.tables))
            
   
        def test_table_link(self):
            t0 = Table()
            self.assertEqual(0, len(t0))
            with self.assertRaises(ValueError):
                t0[5] = 7
            self.assertEqual(0, len(t0))
            t0[5] = 5
            t0[7] = 5
            self.assertEqual(2, len(t0))
            t0[1] = 5
            self.assertEqual(3, len(t0))
            t0[2] = 7
            self.assertEqual(4, len(t0))
            with self.assertRaises(ValueError):
                t0[7] = 9

        def test_cc(self):
            '''
            This is the test case from https://rosalind.info/problems/components/
            '''
            components = Table.create_connected_components([
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
            self.assertEqual(3, len(components))
            self.assertCountEqual([1, 2, 5, 9, 10], components[0])
            self.assertCountEqual([3, 4, 7, 8, 11, 12], components[1])
            self.assertCountEqual([6], components[2])


        def test_verify_consistency1(self):
            table = Table()
            table[3] = 3
            table[2] = 3
            table[1] = 2
            self.assertTrue(table.verify_consistency())

        def test_verify_consistency2(self):
            table = Table()
            table.links[2] = 3
            table.links[1] = 2
            self.assertFalse(table.verify_consistency())

        def test_verify_consistency3(self):
            table = Table()
            table.links[3] = 1
            table.links[2] = 3
            table.links[1] = 2
            self.assertTrue(table.verify_consistency())
            
        def tearDown(self):
            Table.reset()


    main()
