from scipy.sparse import coo_matrix
import heapq as hq
import numpy as np


class graph_model:


    def __init__(self, n_neighbour=4):
        self.graph = None
        self.n_nodes = 0
        self.init_n_neighbour_graph(n_neighbour=n_neighbour)

    def init_n_neighbour_graph(self, n_neighbour=4):
        self.n_neighbour = n_neighbour

    def generate_directed_graph(self, x, w=None, metric='L2', verbose=True):

        self.n_dim   = len(x)
        self.n_nodes = len(x[0])
        if w == None or len(w) != self.n_dim:
            w = np.ones(self.n_dim)

        index_vec = np.array(range(self.n_nodes))

        row = []
        col = []
        wight = []

        distance = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):

             if verbose:
                 if (i%500 == 0): print i

             distance *= 0.0
             for iDim in range(self.n_dim):
                 distance += w[iDim]**2 * (x[iDim] - x[iDim][i])**2
 
             # calculate distance all nodes respect to the central node
             distance[distance < 1e-10] = 1e10
             indexes = hq.nsmallest(self.n_neighbour, index_vec, distance.take)

             # make the graph
             for ind in indexes[:]:
                 
                 row  += [i]
                 col  += [ind]
                 wight += [1]
              
        self.row  = np.array(row, dtype=int)
        self.col  = np.array(col, dtype=int)
        self.wight = np.array(wight)

    def generate_sparse_graph(self):
        return coo_matrix((self.wight, (self.row, self.col)),\
                           shape=(self.n_nodes,self.n_nodes))#.toarray()


    def save_graph(self, fdir='./graphs/'):

        np.savetxt(fdir+'row.txt', self.row, fmt='%i', newline=' ')
        np.savetxt(fdir+'col.txt', self.col, fmt='%i', newline=' ')
        np.savetxt(fdir+'wight.txt', self.wight, newline=' ')

        np.savetxt(fdir+'size.txt', [self.n_nodes,self.n_nodes], fmt='%i')


    def load_graph(self, fdir='./graphs/'): 

        self.row   = np.loadtxt(fdir+'row.txt', dtype=int)
        self.col   = np.loadtxt(fdir+'col.txt', dtype=int)
        self.wight = np.loadtxt(fdir+'wight.txt')

        self.n_nodes = np.loadtxt(fdir+'size.txt', dtype=int)[0]



    def connect_new_node(self, x, x_new, metric='L2'):

        n_dim   = len(x)
        n_nodes = len(x[0])

        index_vec = np.array(range(n_nodes))

        distance = np.zeros(n_nodes)
        for iDim in range(n_dim):
            distance += (x[iDim] - x_new[iDim])**2
 
        # calculate distance all nodes respect to the central node
        indexes = hq.nsmallest(self.n_neighbour, index_vec, distance.take)

        return indexes




class graph_spatial_model:


    def __init__(self, dist=0.1):
        self.graph = None
        self.n_nodes = 0
        self.init_dist_graph(dist=dist)

    def init_dist_graph(self, dist=0.1):
        self.dist = dist

    def generate_directed_graph(self, x, w=None, metric='L2', verbose=True):

        self.n_dim   = len(x)
        self.n_nodes = len(x[0])
        if w == None or len(w) != self.n_dim:
            w = np.ones(self.n_dim)

        index_vec = np.array( range(self.n_nodes) )

        row = []
        col = []
        wight = []

        distance = np.zeros(self.n_nodes)

        for i in range(self.n_nodes):

             if verbose:
                 if (i%500 == 0): print i

             distance *= 0.0
             for iDim in range(self.n_dim):
                 #for iDim in range(4):
                 distance += w[iDim]**2 * (x[iDim] - x[iDim][i])**2
             # calculate distance all nodes respect to the central node
             indexes = index_vec[ ( (distance < self.dist) * (distance > 1e-8) ) ]

             # make the graph
             for ind in indexes[:]:
                 
                 row  += [i]
                 col  += [ind]
                 wight += [1]
              
        self.row  = np.array(row, dtype=int)
        self.col  = np.array(col, dtype=int)
        self.wight = np.array(wight)


    def generate_sparse_graph(self):

        return coo_matrix((self.wight, (self.row, self.col)),\
                           shape=(self.n_nodes,self.n_nodes))#.toarray()


    def save_graph(self, fdir='./graphs/'):

        np.savetxt(fdir+'row.txt', self.row, fmt='%i', newline=' ')
        np.savetxt(fdir+'col.txt', self.col, fmt='%i', newline=' ')
        np.savetxt(fdir+'wight.txt', self.wight, newline=' ')

        np.savetxt(fdir+'size.txt', [self.n_nodes,self.n_nodes], fmt='%i')


    def load_graph(self, fdir='./graphs/'): 

        self.row   = np.loadtxt(fdir+'row.txt', dtype=int)
        self.col   = np.loadtxt(fdir+'col.txt', dtype=int)
        self.wight = np.loadtxt(fdir+'wight.txt')

        self.n_nodes = np.loadtxt(fdir+'size.txt', dtype=int)[0]


































# test
#cw = cosmic_graph()

#x = np.array([1,2,3,4,5,6,7])     
#y = np.array([1,2,3,4,5,6,7])     
#z = np.array([1,2,3,4,5,6,7]) 

#cw.generate_directed_graph(x, y, z)
#print cw.generate_sparse_graph().toarray()
#cw.save_graph(fdir='./graphs/')

#cw_new = cosmic_graph()
#cw_new.load_graph(fdir='./graphs/')
#print cw_new.generate_sparse_graph().toarray()


"""
>>> from scipy.sparse import coo_matrix
>>> coo_matrix((3, 4), dtype=np.int8).toarray()
array([[0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]], dtype=int8)
>>>
>>> row  = np.array([0, 3, 1, 0])
>>> col  = np.array([0, 3, 1, 2])
>>> data = np.array([4, 5, 7, 9])
>>> coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
array([[4, 0, 9, 0],
       [0, 7, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 5]])
"""
