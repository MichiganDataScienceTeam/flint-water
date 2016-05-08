import numpy as np
import sys
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
import scipy.io as scio
import pandas as pd    
import scipy.stats




class density_pipeline():


    def __init__(self):

        from model import density_map
        from model import sparse_density_map
 
        self.dmap = sparse_density_map()
        #self.graph = graph_spatial_model( dist=0.0002 )
        self.data = pd.read_csv('./data/lead_parcel_data_train.csv')
        self.clean_dataset()


    def clean_dataset(self):

        for iLabel in ['Latitude','Longitude']:
            self.data[iLabel] = self.data[iLabel].fillna(self.data[iLabel].iloc[0])

        self.data = self.data[np.array(self.data['Year Built'] > 10)]
        #self.data = self.data[np.array(self.data['Lead (ppb)'] > 0.1)]
            

    def generate_density_map(self, npix=100):


        # generate network

        L1   = np.array(self.data['Latitude']) 
        L2   = np.array(self.data['Longitude']) 
        Lead = np.log10(np.array(self.data['Lead (ppb)']) + 1.0)
        #Lead = np.array(self.data['Lead (ppb)']) 

        #PropClass = np.array(self.data['Prop Class'].apply(hash)%4) / 5.
        #PA = np.array((self.data['Parcel Acres']+1).apply(np.log10))
        #PA *= .01
        #print min(PA), max(PA)
        #Rental = np.array(self.data['Rental'].apply(hash)%2) / 20.
        #HydrantType = np.array( self.data['Hydrant Type'].apply(hash) / 2.)
        year = np.array(self.data['Year Built'])

        self.dmap.generate_map(L1, L2, Lead, npix=npix)
        self.dmap.save_map(fdir='./output/graph/model_')



    def analyse_density_map(self): 

        dimg, cimg = self.dmap.generate_density_map(filter='Gaussian',sigma=1.)

        self.dmap.make_plot(dimg, vmin=0.2, vmax=1.3)
        #self.dmap.make_plot(self.dmap.map / self.dmap.counter, vmin=1900, vmax=1990)
        plt.show()
         
        self.dmap.make_plot(cimg)
        plt.show()


    def animation_density_map(self):

        from matplotlib import pyplot as plt
        from matplotlib import animation

        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15,metadata=dict(artist='Arya Farahi'),bitrate=14400)

        # First set up the figure,
        # the axis, and the plot element we want to animate
        fig = plt.figure()
        #ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
 
        def updated_map(ipix,sigma):
            self.generate_density_map(npix=ipix)
            dimg, cimg = self.dmap.generate_density_map(filter='Gaussian',\
                                   sigma=sigma, mask_nan=False)
            return dimg #self.dmap.counter

        im = plt.imshow(updated_map(40,0.1), \
                     interpolation='none', origin='lower',\
                     cmap='hot', animated=True, vmin=0.1, vmax=1.3)
        # lead: vmin=0.1, vmax=1.3)
        # counter: vmin=0, vmax=10) 
        # year: vmin=1920, vmax=1990

        plt.colorbar()

        def animate(i):
            ipix = 40 + i #40 + i*5
            sigma = 0.5 + i*0.1 #ipix**2 / float(40**2 * 10)
            if sigma > 1.5: sigma = 1.5
            print ipix, sigma, i
            im.set_array(updated_map(ipix,sigma))
            #im.set_array(updated_map(200,0.2*i+0.1))
            return im,

        # call the animator.
        # blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate,
                               frames=160, interval=20, blit=False)

        anim.save('./output/animation/log-lead.mp4', writer=writer)
        #plt.show()


"""
fig = plt.figure()


def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

im = plt.imshow(f(x, y), cmap='hot', animated=True)


def updatefig(*args):
    global x, y
    x += np.pi / 15.
    y += np.pi / 20.
    im.set_array(f(x, y))
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()
"""
        

cols = [ u'Sample Number', u'Date Submitted', u'Lead (ppb)', u'Copper (ppb)',
         u'Street #', u'Street Name', u'City', u'Zip Code', u'Best Address',
         u'PID Dash', u'PID no Dash', u'Property Address', u'Property Zip Code',
         u'Owner Type', u'Owner Name', u'Owner Address', u'Owner Zip Code',
         u'Owner City', u'Owner State', u'Owner Country', u'Tax Payer Name',
         u'Tax Payer Address', u'Tax Payer State', u'Tax Payer Zip Code',
         u'Homestead', u'Homestead Percent', u'HomeSEV', u'Land Value',
         u'Land Improvements Value', u'Residential Building Value',
         u'Residential Building Style', u'Commercial Building Value',
         u'Building Storeys', u'Parcel Acres', u'Rental', u'Use Type',
         u'Prop Class', u'Old Prop class', u'Year Built', u'USPS Vacancy',
         u'Zoning', u'Future Landuse', u'DRAFT Zone', u'Housing Condition 2012',
         u'Housing Condition 2014', u'Commercial Condition 2013', u'Latitude',
         u'Longitude', u'Hydrant Type' ]




"""
import numpy as np
import cosmicGraph as cg

bolshoi = '/home/aryaf/Pipelines/spectroscopicModel/bolshoiSimulation/Mr19_age_distribution_matching_mock.dat'

halo_id, x, y, z, vx, vy, vz, Mvir, Vpeak, m_r, g_r, Mhost, host_halo_id = \
        np.loadtxt(bolshoi, unpack=True)

print len(x)


cw = cg.cosmic_graph()

#x = np.array([1,2,3,4,5,6,7])     
#y = np.array([1,2,3,4,5,6,7])     
#z = np.array([1,2,3,4,5,6,7]) 



cw.generate_directed_graph(x, y, z)
cw.save_graph(fdir='./graphs/bolshoi_9_')

#print cw.generate_sparse_graph().toarray()[:10,:10]

    if stepNum == 2: pass
    if stepNum == 3: pass
    if stepNum == 4: pass
"""


