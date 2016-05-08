from scipy.sparse import coo_matrix
import matplotlib.pylab as plt
from scipy import ndimage
import heapq as hq
import numpy as np


class density_map:

    def __init__(self):
        self.map = None

    def generate_map(self, x, y, value, xrange=None, yrange=None, npix=10):

        def find_index(x,xmin,dx):
            return int((x-xmin)/dx)
 
        
        self.npix    = npix
        self.npoints = len(x)
        self.background = np.median(value)

        if xrange == None:
            self.xrange = (min(x), max(x))
        if yrange == None:
            self.yrange = (min(y), max(y))

        dx = (self.xrange[1] - self.xrange[0]) / float(npix)
        dy = (self.yrange[1] - self.yrange[0]) / float(npix)

        self.map     = np.zeros([npix,npix]) 
        self.counter = np.zeros([npix,npix])

        for ix, iy, ival in zip(x,y,value):
            # calculate index
            i = find_index(ix,self.xrange[0],dx)
            j = find_index(iy,self.yrange[0],dy)

            # check whther index is inside the boundary
            if i < 0: continue
            if j < 0: continue
            if i > npix: continue
            if j > npix: continue
            if i == npix: i -= 1 
            if j == npix: j -= 1 

            # add the point to the map
            self.map[i,j] += ival
            self.counter[i,j] += 1
            

    def generate_simple_density_map(self):
        if map == None: 
            print "Error: Generate map first"
            return 0
        density_map = self.map / self.counter 
        #density_map[np.isnan(density_map)] = 0.
        return density_map, np.isnan(density_map)

    def generate_density_map(self,filter='Gaussian',sigma=1,order=3,\
                                  mask_nan=False):

        dmap, nanmask = self.generate_simple_density_map()
        dmap[nanmask] = self.background

        if filter == 'Gaussian':
            dmap = ndimage.filters.gaussian_filter(dmap, sigma=sigma)
            dcount = ndimage.filters.gaussian_filter(self.counter, sigma=sigma)
        if filter == 'Uniform':
            dmap = ndimage.filters.uniform_filter(dmap, size=sigma)
            dcount = ndimage.filters.uniform_filter(self.counter, size=sigma)
        if filter == 'Median':
            dmap = ndimage.filters.median_filter(dmap, size=sigma)
            dcount = ndimage.filters.median_filter(self.counter, size=sigma)
        if filter == 'Spline':
            dmap = ndimage.filters.spline_filter(dmap, order=order)
            dcount = ndimage.filters.spline_filter(self.counter, order=order)
        
        if mask_nan:    
            dmap[nanmask] = np.nan

        return dmap, dcount


    def make_plot(self, dmap, cmap='hot', **kwargs):

        plt.imshow(dmap, extent=(self.xrange[0], self.xrange[1], \
                                 self.yrange[0], self.yrange[1]), \
                     interpolation='none', origin='lower',\
                     cmap=cmap, **kwargs)
        plt.colorbar()
    


    def save_map(self, fdir='./output/density_map/'):

        if map == None: 
            print "Error: Generate map first"
        else:
            np.savetxt(fdir+'map.txt', self.map)
            np.savetxt(fdir+'counter.txt', self.counter, fmt='%i')


    def load_graph(self, fdir='./output/density_map/'): 

        self.map     = np.loadtxt(fdir+'map.txt')
        self.counter = np.loadtxt(fdir+'counter.txt', dtype=int)





class sparse_density_map:

    def __init__(self):
        self.map = None

    def generate_map(self, x, y, value, xrange=None, yrange=None, npix=10):

        def find_index(x,xmin,dx):
            return int((x-xmin)/dx)
 
        
        self.npix    = npix
        self.npoints = len(x)
        self.background = np.median(value)

        if xrange == None:
            self.xrange = (min(x), max(x))
        if yrange == None:
            self.yrange = (min(y), max(y))

        dx = (self.xrange[1] - self.xrange[0]) / float(npix)
        dy = (self.yrange[1] - self.yrange[0]) / float(npix)

        self.map     = np.zeros([npix,npix]) 
        self.counter = np.zeros([npix,npix])

        for ix, iy, ival in zip(x,y,value):
            # calculate index
            i = find_index(ix,self.xrange[0],dx)
            j = find_index(iy,self.yrange[0],dy)

            # check whther index is inside the boundary
            if i < 0: continue
            if j < 0: continue
            if i > npix: continue
            if j > npix: continue
            if i == npix: i -= 1 
            if j == npix: j -= 1 

            # add the point to the map
            self.map[i,j] += ival
            self.counter[i,j] += 1
            

    def generate_simple_density_map(self):
        if map == None: 
            print "Error: Generate map first"
            return 0
        density_map = self.map / self.counter 
        #density_map[np.isnan(density_map)] = 0.
        return density_map, np.isnan(density_map)

    def _fill_the_blank(self, dmap, nanmask):
       
        filled = {'x':[], 'y':[], 'val':[], 'count':[]}
        blank  = {'x':[], 'y':[], 'val':[]}
        for i in range(self.npix):
            for j in range(self.npix):
                if nanmask[i,j]:
                    blank['x']   += [i] 
                    blank['y']   += [j] 
                else:
                    filled['x']   += [i] 
                    filled['y']   += [j] 
                    filled['val'] += [dmap[i,j]] 
                    filled['count'] += [self.counter[i,j]] 

        for iLable in ['x','y','val','count']:
            filled[iLable] = np.array(filled[iLable])

        distance  = np.zeros(len(filled['x']))     
        index_vec = np.array(range(len(filled['x'])))

        for i, j in zip(blank['x'], blank['y']):
            distance *= 0
           
            distance += (filled['x'] - i)**2
            distance += (filled['y'] - j)**2

            indexes = hq.nsmallest(12, index_vec, distance.take)

            dmap[i,j] = np.average(filled['val'][indexes],\
                              weights=filled['count'][indexes]/distance[indexes])

        return dmap



    def generate_density_map(self,filter='Gaussian',sigma=1,order=3,\
                                  mask_nan=False):

        dmap, nanmask = self.generate_simple_density_map()
        dmap = self._fill_the_blank(dmap, nanmask)
        #dmap[nanmask] = self.background

        if filter == 'Gaussian':
            dmap = ndimage.filters.gaussian_filter(dmap, sigma=sigma)
            dcount = ndimage.filters.gaussian_filter(self.counter, sigma=sigma)
        if filter == 'Uniform':
            dmap = ndimage.filters.uniform_filter(dmap, size=sigma)
            dcount = ndimage.filters.uniform_filter(self.counter, size=sigma)
        if filter == 'Median':
            dmap = ndimage.filters.median_filter(dmap, size=sigma)
            dcount = ndimage.filters.median_filter(self.counter, size=sigma)
        if filter == 'Spline':
            dmap = ndimage.filters.spline_filter(dmap, order=order)
            dcount = ndimage.filters.spline_filter(self.counter, order=order)
        
        if mask_nan:    
            dmap[nanmask] = np.nan

        return dmap, dcount


    def make_plot(self, dmap, cmap='hot', **kwargs):

        plt.imshow(dmap, extent=(self.xrange[0], self.xrange[1], \
                                 self.yrange[0], self.yrange[1]), \
                     interpolation='none', origin='lower',\
                     cmap=cmap, **kwargs)
        plt.colorbar()
    


    def save_map(self, fdir='./output/density_map/'):

        if map == None: 
            print "Error: Generate map first"
        else:
            np.savetxt(fdir+'map.txt', self.map)
            np.savetxt(fdir+'counter.txt', self.counter, fmt='%i')


    def load_graph(self, fdir='./output/density_map/'): 

        self.map     = np.loadtxt(fdir+'map.txt')
        self.counter = np.loadtxt(fdir+'counter.txt', dtype=int)













# test

#import numpy.random as npr
#dm = density_map()
#sdm = sparse_density_map()

#npr.seed(1)
#x = npr.random(5000) #np.array([1,2,3,4,5,6,7])     
#y = npr.random(5000) #np.array([1,2,3,4,5,6,7])     
#z = npr.random(5000) #np.array([1,2,3,4,5,6,7]) 

#dm.generate_map(x, y, z, npix=200)
#dmap, cmap = dm.generate_density_map(filter='Gaussian',sigma=0.001)
#dm.make_plot(dmap)
#plt.show()

#sdm.generate_map(x, y, z, npix=200)
#dmap, cmap = sdm.generate_density_map(filter='Gaussian',sigma=0.001)
#sdm.make_plot(dmap)
#plt.show()


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
