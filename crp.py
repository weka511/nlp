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

__all__ = ['Table', 'DistanceDependentChooser', 'NoDecay']
__version__ = '0.0'
__author__ = 'Simon Crase'

from abc import ABC, abstractmethod
from pathlib import Path
from unittest import TestCase, main, skip
import numpy as np
from shared.utils import Logger

type Node = int            # A node number
type NNode = int           # The number of nodes
type Link = tuple[Node,Node]

class TableOwner(ABC):
    '''
    Abstract base class for Chinese Restaurant process. It encapsulates
    the knoweldge that every non-empty table must belong to some node.
    
    '''
    def __init__(self, chooser:'Chooser' = None):
        '''
        Parameters:
            chooser
        '''
        self.m = chooser.get_m()
        self.tables = np.empty((self.m), dtype=Table)
        self.chooser = chooser
        
    def __getitem__(self,node : Node):
        '''
        Find out which table contains specified node
        
        Parameters:
            node
        '''
        return self.tables[node]
    
    def __setitem__(self,node : Node, table: 'Table'):
        '''
        Record the table that contains spcified node
        
        Parameters:
            node
            table
        '''
        self.tables[node] = table
     
    def generate_tables(self):
        '''
        Used to iterate through all tables that are managed by this owner
        '''
        table_seqs = set()
        for node in range(self.m):
            table = self.tables[node]
            if table.seq not in table_seqs:
                yield table
                table_seqs.add(table.seq)
    
    def verify_tables(self):
        '''
        Used to verify that all tables are linked correctly in self.tables
        '''
        for table in self.generate_tables():
            self.verify_linked_correctly(table)
        
    def verify_linked_correctly(self,table):
        '''
        Used to verify that specified table is linked correctly in self.tables
        
        Parameters:
            table
        '''        
        for start,end in table.links.items():
            assert self.tables[start] == table
            assert self.tables[end] == table    
                
        
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

    def __init__(self):
        '''
        Initialize links to an empty list, allocate a sequence number to the table,
        and store newly created table in a list of tables.
        '''
        self.links = {}
        self.seq = len(Table.tables)
        Table.tables.append(self)

    def __eq__(self, table : 'Table'):
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

    def __getitem__(self, key : Node) -> Node:
        '''
        Get the node that the supplied node links to
        
        Parameters:
            key        Start of link
        '''
        return self.links[key]

    def __setitem__(self, start : Node, end : Node):
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

    def break_at(self, node : Node,owner : TableOwner) -> 'Table':
        '''
        Break one link in a Table.
        
        Parameters:
            node      The node whose link is to be broken
            owner     Used to update map from nodes to tables
            
        Returns:
           The table which contains `node': this may be a newly created table, or the original
        '''

        # Simulate removal of link, which may split Table into two components
        components = Table.create_connected_components(self.create_edge_list(omit=node))
        
        match len(components):       # Will table be split?
            
            case 0:       # This should never happen, unless table had only one element
                Logger.get_instance().log(f'{__file__} {Logger.get_line()} there are no components\n'
                                          f'Table length = {len(self)}',
                                          level=Logger.WARNING)
                Logger.get_instance().log(f'{__file__} {Logger.get_line()} {self}',
                                          level=Logger.WARNING)                
                return self

            case 1:                          # Break must be in a cycle: we will have only one table
                Logger.get_instance().log(f'{__file__} {Logger.get_line()} Only one component',
                                          level=Logger.DEBUG)
                self.links[node] = node      # Simply make node point to itself
                owner.verify_tables()
                return self

            case 2:                               # Break will split links in two
                Logger.get_instance().log(f'{__file__} {Logger.get_line()} Break will split links in two ')
                new_table, to_delete = self._move_nodes(node,self._standardize(node,components),owner)
                Logger.get_instance().log(f'{__file__} {Logger.get_line()} len={len(self)}, len_new={len(new_table)}'
                                          f', len_to_delete={len(to_delete)}')
                self._remove_redundant_links(to_delete)
                Logger.get_instance().log(f'{__file__} {Logger.get_line()} len={len(self)}, len_new={len(new_table)}'
                                          f', len_to_delete={len(to_delete)}')
                Logger.get_instance().log(f'{__file__} {Logger.get_line()} {self} ',level=Logger.DEBUG)
                Logger.get_instance().log(f'{__file__} {Logger.get_line()} {new_table} ',level=Logger.DEBUG)
                owner.verify_tables()
                return new_table

            case _:
                raise RuntimeError(f'There are {len(components)} connected components')
            
    def _standardize(self,node,components):
        '''
            Used when we are breaking a link, to ensure that the node is in the first component.
            
            Parameters:
                node         The node where we are breaking
                components   A list of nodes, partitioned into those that belong with node, and the rest
        '''
        if node in components[1]: return components
        
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Swapping components')
        return [components[1], components[0]]

    def _move_nodes(self,node,components,owner):
        '''
        Used by break_at() to move the nodes, and make a list of those those whose
        links will be deleted from the original table
        
        Parameters:
            node         The node where we are breaking
            components   A list of nodes, partitioned into those that belong with node, and the rest
            
        Returns:
            The original table (if nothing moved), or a new one with those nodes that have moved
            A list of nodes that were moved, which should be deleted from original
        '''
        new_table = Table()
        new_table[node] = node
        owner[node] = new_table
        to_delete = set()
        if not node in components[1]:
            Logger.get_instance().log(f'{__file__} {Logger.get_line()} {node} {components[1]}',level=Logger.WARNING)
        for a, b in self.links.items():
            if a in components[0] and b in components[0]:
                pass # keep link
            elif a in components[1] and b in components[1]:
                try:
                    new_table[a] = b
                except ValueError:
                    Logger.get_instance().log(f'{__file__} {Logger.get_line()} a={a},b={b},old={new_table[a]}')
                    
                to_delete.add(a)
                to_delete.add(b)
                owner[a] = new_table
            elif a != node:  # Ignore split if origin is node; we expect that case to split
                Logger.get_instance().log(f'{__file__} {Logger.get_line()} a={a},b={b} separated',level=Logger.WARNING)

        return new_table, to_delete   

    def _remove_redundant_links(self,to_delete):
        '''
        Used by break_at() to remove links for those nodes that were removed
        
        Parameters:
            to_delete
        '''
        for a in to_delete:
            del self.links[a]

    def join(self, node : Node, table_to_join : 'Table', node_to_connect : 'Node', owner:TableOwner,purge=True):
        '''
        Join two groups of nodes
        
        Parameters:
            node             The node that we are going to connect to
            table_to_join    Contains nodes that are to be joined to this table
            node_to_connect  The node that is being linked
            owner            The Chinese Restaurant Process, which keeps a record of which table items belong in
            purge            Used when table are disticct to purge table that has been joined
        '''
        self.links |= table_to_join.links
        for a,b in table_to_join.links.items():
            owner[a] = self
            owner[b] = self
        self.links[node_to_connect] = node
        owner[node_to_connect] = self
        if purge:
            table_to_join.links.clear()
        owner.verify_tables()

    def create_edge_list(self, omit : Node =-1) -> list[Link]:
        '''
        Convert links to edge-link format so we can use create_cc, 
        optionally omitting one specified link
        
        Parameters:
             omit      Link to be omitted
        '''
        return [(start, end) for start, end in self.links.items() if start != omit and start != end]

    def generate_links(self):
        '''
        Used to iterate through all nodes in Table
        '''
        for start, end in self.links.items():
            yield start, end

    def verify_consistency(self,line=None):
        '''
        Verify (1) that Table has one component only; (2) every node goes somewhere
        '''
        def table_has_one_component_only():
 
            #if self.seq == 12:
                #z=0
            components = Table.create_connected_components(self.create_edge_list())
            match len(components):
                case 0:
                    if len(self) == 1: return True
                    Logger.get_instance().log(f'{__file__} {Logger.get_line()} There are {len(components)} components',
                                              level=Logger.WARNING)
                    Logger.get_instance().log(f'{__file__} {Logger.get_line()} {self} len={len(self)}',
                                              level=Logger.WARNING)                    
                    return False                
                case 1:
                    return True
                case _:
                    Logger.get_instance().log(f'{__file__} {Logger.get_line()} There are {len(components)} components',
                                              level=Logger.WARNING)
                    return False

        def every_node_goes_somewhere():
            starts = set(self.links.keys())
            ends = set(self.links.values())
            unterminated = ends - starts
            if len(unterminated) == 0:
                return True
            else:
                Logger.get_instance().log(f'{__file__} {Logger.get_line()} There are {len(unterminated)} unterminated',
                                          level=Logger.WARNING)
                Logger.get_instance().log(f'{__file__} {Logger.get_line()} {unterminated}',
                                          level=Logger.WARNING)
                #Logger.get_instance().log(f'{__file__} {Logger.get_line()} {self.links}',
                                          #level=Logger.WARNING)                   
                return False
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} {line}',
                                  level=Logger.INFO)
        return table_has_one_component_only() and every_node_goes_somewhere()

    @staticmethod
    def create_connected_components(g : list[Link]):
        '''
        Create a list of connected components for a graph
        
        Parameters:
            g       Graph in edge list format
        '''
        def create_vertices(g):
            '''
            Construct the set of all vertices in the graph
            
            Parameters:
                g       Graph in edge list format
                
            Returns:
                Set of vertices
            '''
            product = set()
            for start, end in g:
                product.add(start)
                product.add(end)
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
            This is the heart of create_connected_components(). It performs Depth First Search 
            to visit every node in a component once and only once.
            
            Parameters:
                vertex         A node that has not been visited yet
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
                    component = component | dfs(child, g_augmented, visited)
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
    def choose(self, n:Node, initial=False) -> Node:
        '''
        Choose which node to link to
        '''

    @abstractmethod
    def get_m(self) -> NNode:
        '''
        Used to tell ChinesRestaurantProcess how many nodes it can expect
        '''
        ...


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

    def get_m(self) -> NNode:
        return self.m

    def get_indices(self, m:Node):                #FIXME is this used anywhere?
        return self.rng.permutation(m)

    def choose(self, n:NNode, initial:bool=False) -> Node:
        '''
        Choose which node to link to
        
        Parameters:
            n 
        '''
        return self.rng.choice(n + 1 if initial else self.m, p=self._get_p(n, initial=initial))

    def _get_p(self, node:NNode, initial:bool=False):
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


class ChineseRestaurantProcess(TableOwner):
    '''
    This class represents a Chinese Restaurant Process. 
    
    Attributes:
        tables   Lookup table to find which Table holds each node
    '''

    UNASSIGNED = -1

    def __init__(self, chooser:'Chooser'=None):
        '''
        Parameters:
            chooser
        '''
        super().__init__(chooser)

    def build(self):
        '''
        Allocate each node to a table. It randomizes the order of nodes, then adds
        one at a time. If the 
        '''
        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Building for {self.m} nodes')
        for this_node in range(self.m):
            link_to = self.chooser.choose(this_node, initial=True)
            if this_node == link_to:
                self._link(this_node, link_to, table=Table())
            else:
                self._link(this_node, link_to, table=self.tables[link_to])

        assert self.m == sum(len(table) for table in Table.tables)

        #for table in Table.generate_tables():
            #table.verify_consistency()
            
        Logger.get_instance().log(f'{__file__} Complete')

    def gibbs(self):
        '''
        Perform one gibbs step
        '''
        for node in range(self.m):
            for table in self.tables:
                if len(table) ==0:
                    Logger.get_instance().log(f'{__file__} {Logger.get_line()} {table} has zero length')
            Logger.get_instance().log(f'{__file__} {Logger.get_line()} Processing Node {node}')
            table_where_node_lives = self.tables[node]
            if not table_where_node_lives.verify_consistency(Logger.get_line()):
                Logger.get_instance().log(f'{__file__} {Logger.get_line()} Inconsistent')
                
            # Break link from node
            table_after_break = table_where_node_lives.break_at(node,self)
            if not table_after_break.verify_consistency(Logger.get_line()):
                Logger.get_instance().log(f'{__file__} {Logger.get_line()} inconsistent')
            link_to = self.chooser.choose(node)
            table_to_link_to = self.tables[link_to]
            #while table_to_link_to == table_after_break:  #FIXME #66 arises whn this equality is true
                #link_to = self.chooser.choose(node)
                #table_to_link_to = self.tables[link_to]                
            Logger.get_instance().log(f'{__file__} {Logger.get_line()} link to {link_to} in table {table_to_link_to.seq}, len= {len(table_to_link_to)}')
            table_to_link_to.verify_consistency(Logger.get_line())
            
            # Now link node to some other node (or itself)
            
            if table_where_node_lives == table_after_break:
                if table_where_node_lives == table_to_link_to:
                    table_where_node_lives.links[node] = link_to
                    if not table_where_node_lives.verify_consistency(Logger.get_line()):
                        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Inconsistent')
                else:
                    table_to_link_to.join(link_to, table_where_node_lives, node,self)
                    if not table_to_link_to.verify_consistency(Logger.get_line()):
                        Logger.get_instance().log(f'{__file__} {Logger.get_line()} Inconsistent')
            else:
                #ll = len(table_to_link_to)
                #seq = table_to_link_to.seq
                #assert table_to_link_to != table_after_break
                table_to_link_to.join(link_to, table_after_break, node,self,purge=table_to_link_to != table_after_break)
                #Logger.get_instance().log(f'{__file__} {Logger.get_line()} {ll} {seq} {len(table_to_link_to)}')
                if not table_to_link_to.verify_consistency(Logger.get_line()):
                    Logger.get_instance().log(f'{__file__} {Logger.get_line()} Inconsistent')
 
            self.tables[node] = table_to_link_to
            components = Table.create_connected_components(table_to_link_to.create_edge_list())
            if len(components) != 1:
                Logger.get_instance().log(f'{__file__} {Logger.get_line()} {len(components)} components')

    def _link(self, start : Node, end : Node, table : 'Table'=None):
        '''
        Link two nodes, and also record that the first node is in the table
        
        Parameters:
            start    This node is to be linked to other node
            end      This is the node to link to
            table    The link lives in this table
        '''
        table[start] = end
        self[start] = table


if __name__ == '__main__':
    class MockChooser(Chooser):
        '''
        This class allows tests to build Tables
        '''

        def __init__(self, nodes):
            self.nodes = nodes
            self.pos = -1

        def choose(self, n, initial=False):
            self.pos += 1
            return self.nodes[self.pos]

        def get_m(self):
            return len(self.nodes)

    class MockOwner(TableOwner):
        def __init__(self):
            super().__init__(MockChooser([]))

    class TestTable(TestCase):
        '''
        Parent for test cases: ensure that all tests tearDown
        '''

        def setUp(self):
            self.logger = Logger(Path(__file__).stem, path='./logs', level=Logger.ERROR)
            self.logger.__enter__()

        def tearDown(self):
            '''
            Purge list of all Tables to prevent test classes interacting
            '''
            Table.tables.clear()
            self.logger.__exit__(None, None, None)

    class TestFigure3(TestTable):
        '''
        Tests based on Figure 3 of Blei & Frazier
        '''

        def setUp(self):
            super().setUp()
            Table.tables.clear()
            chooser = MockChooser([0, 0, 1, 2, 4, 4])
            self.crp = ChineseRestaurantProcess(chooser=chooser)
            self.crp.build()
            
        def test_verify_tables(self):
            self.crp.verify_tables()

        def test_build_tables(self):
            '''
            This test verifies that we can construct the graph in the first panel of Figure 3 of Blei & Frazier
            '''
            self.assertEqual(2, len(Table.tables))
            table0 = Table.tables[0]
            table1 = Table.tables[1]
            self.assertNotEqual(table0, table1)
            self.assertEqual(0, table0[0])
            self.assertEqual(0, table0[1])
            self.assertEqual(1, table0[2])
            self.assertEqual(2, table0[3])
            self.assertEqual(4, table1[4])
            self.assertEqual(4, table1[5])
  
        def test_break_link(self):
            '''
            This test verifies that we can construct the graph in lines 2 and 3 of Figure 3 of Blei & Frazier
            '''
            table = Table.tables[0]
            table_split = table.break_at(2,self.crp)
            self.assertNotEqual(table_split, table)
            self.assertEqual(3, len(Table.tables))
            table_join_to = self.crp.tables[4]
            table_join_to.join(4, table_split, 2,self.crp)
            self.assertEqual(6, sum([len(t) for t in Table.tables]))
            self.assertEqual(4, len(table_join_to))
            self.assertEqual(0, Table.tables[0][0])
            self.assertEqual(0, Table.tables[0][1])
            self.assertEqual(4, Table.tables[1][4])
            self.assertEqual(4, Table.tables[1][5])
            self.assertEqual(4, Table.tables[1][2])
            self.assertEqual(2, Table.tables[1][3])

 
    class TestCycle(TestTable):
        '''
        Test case for tables with non-trivial cycles
        '''
        
        def test_break_cycle(self):
            '''
            Break a simple cycle so it becomes 1D
            '''
            table = Table()
            owner = MockOwner()
            table.links = {1: 0, 2: 1, 3: 2, 4: 3, 0: 4}
            self.assertEqual(table, table.break_at(3,owner))
            self.assertEqual(0, table.links[1])
            self.assertEqual(1, table.links[2])
            self.assertEqual(3, table.links[3])
            self.assertEqual(3, table.links[4])
            self.assertEqual(4, table.links[0])

    class TestSimple(TestTable):

        def test_table_create(self):
            t0 = Table()
            self.assertEqual(0, t0.seq)
            t1 = Table()
            self.assertEqual(1, t1.seq)
            self.assertNotEqual(t0, t1)
            self.assertEqual(2, len(Table.tables))

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

    main()
