import numpy as np

class analysis_n_neighbour_graph:

    def __init__(self,n_neighbour):
        self.n_neighbour = n_neighbour


    def mean_neighbour_value(self, row, col, wight, n_nodes, node_vec):
        # mean neighbour value
        mnv = np.zeros(n_nodes)
        neighbours = np.zeros(n_nodes)

        for irow, icol, iwight in zip(row, col, wight):

            mnv[irow] += iwight*node_vec[icol]
            #mnv[icol] += iwight*node_vec[irow]
            neighbours[irow] += iwight
         
        return mnv / neighbours


    def mean_neighbour(self, row, col, wight, n_nodes):

        # mean and std neighbour 
        neighbours = np.zeros(n_nodes)

        for irow, icol, iwight in zip(row, col, wight):
            neighbours[irow] += iwight

        return np.mean(neighbours), np.std(neighbours)



"""
import cosmicGraph as cg
import numpy.random as npr
npr.seed(200)

cw = cg.cosmic_graph()
graph_analysis = analysis_n_neighbour_graph(4)

#x = np.array([1,2,3,4,5,6,7])     
#y = np.array([1,2,3,4,5,6,7])     
#z = np.array([1,2,3,4,5,6,7]) 
#data = np.array([1,4,5,3,2,4,5])

x = npr.random(500)  
y = npr.random(500)
z = npr.random(500)
data = npr.random(500)

cw.generate_directed_graph(x, y, z)


mean_n_data = graph_analysis.mean_neighbour_value(cw.row, cw.col, cw.wight, cw.n_nodes, data)

import scipy.stats

print scipy.stats.pearsonr(data, mean_n_data)
"""

