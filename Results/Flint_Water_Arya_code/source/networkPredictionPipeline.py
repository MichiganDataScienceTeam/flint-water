import numpy as np
import sys
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
import scipy.io as scio
import pandas as pd    
import scipy.stats



class network_pipeline():


    def __init__(self, n_neighbour=20, dist=0.0002):

        from model import graph_model, graph_spatial_model

        self.n_neighbour = n_neighbour 
        self.graph = graph_model( n_neighbour=n_neighbour )
        #self.graph = graph_spatial_model( dist=0.0002 )
        self.data = pd.read_csv('./data/sentinel_test_data.csv')
        self.clean_dataset()

        self.dic_nodes= {'x':[],'y':[], 'ind':[]}


    def clean_dataset(self):

        for iLabel in ['Latitude','Longitude']:
            self.data[iLabel] = self.data[iLabel].fillna(self.data[iLabel].iloc[0])
        self.data = self.data[np.array(self.data['Year Built'] > 10)]
        #self.data = self.data[np.array(self.data['Lead (ppb)'] > 0.1)]
        #self.data = self.data[np.array(self.data['Copper (ppb)'] > 0.1)]
        #self.data = self.data[np.array(self.data['Use Type'] == 'Residential' )]
        print "Length data :", len(self.data) 


    def generate_network(self):

        def generate_nodes(x,y):

            data = pd.read_csv('./data/sentinel_test_data.csv')
           
            Long = np.array(data['Longitude'])
            Lat  = np.array(data['Latitude'])

            return Long[::], Lat[::]

            """
            import numpy.random as npr

            x_min = min(x)
            x_max = max(x)
            y_min = min(y)
            y_max = max(y)
 
            x_nodes = npr.uniform(x_min,x_max,1000)
            y_nodes = npr.uniform(y_min,y_max,1000)
                 
            return x_nodes, y_nodes
            """

        def return_value():
            
            #return self.data['Lead (ppb)']
            return np.log10(self.data['Lead (ppb)'] + 1.)


        # generate network
        Lat = np.array(self.data['Latitude']) 
        Long = np.array(self.data['Longitude']) 
        sub_nodes = np.array([Long,Lat])
        value = return_value()

        x_nodes , y_nodes = generate_nodes(Long,Lat) 
        self.dic_nodes= {'x':[],'y':[], 'ind':[], 'value':[]}

        for ix, iy in zip(x_nodes, y_nodes):

            self.dic_nodes['x'] += [ix]
            self.dic_nodes['y'] += [iy]
            indexes = self.graph.connect_new_node(sub_nodes, [ix,iy])
            self.dic_nodes['ind'] += [indexes]
            self.dic_nodes['value'] += [np.mean(value[indexes])]


    def _generate_new_value(self,tag='Year Built'):
          
        def return_value(tag):
            return np.array(self.data[tag])

        val = return_value(tag)

        new_value = np.zeros(len(self.dic_nodes['x']))
        for i in range(len(new_value)):
            indexes =  self.dic_nodes['ind'][i]
            new_value[i] = np.mean(val[indexes])

        return new_value  


    def realize_network(self):

        import matplotlib.colors as colors
        import matplotlib.cm as cmx
        """
        #colorMap = plt.get_cmap('seismic')
        colorMap = plt.get_cmap('cool')
        cNorm = colors.Normalize(vmin=1920, vmax=1990)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colorMap)

        x = self.dic_nodes['x']; y = self.dic_nodes['y'] 
        sizeValue = np.array( self.dic_nodes['value'] )
        colorValue = self._generate_new_value(tag='Year Built')

        for i in range(len(x)):       
            colorVal = scalarMap.to_rgba(colorValue[i])
            plt.scatter(x[i],y[i],s=sizeValue[i],color=colorVal)

        plt.show()    
        #plt.savefig('./output/graph_vis/Lead.png', bbox_inches='tight')
        """

    def over_plot_googlemap(self):
 
        import folium
        from folium import plugins
        import matplotlib.colors as colors
        import matplotlib.cm as cmx

        colorMap = plt.get_cmap('cool')
        cNorm = colors.Normalize(vmin=1920, vmax=1990)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colorMap)

        x = self.dic_nodes['x']; y = self.dic_nodes['y'] 
        Lead = 10**np.array( self.dic_nodes['value'] ) - 1.
        Year = self._generate_new_value(tag='Year Built')

        map = folium.Map(location=[43.0125, -83.6875], zoom_start=13)

        for i in range(len(x)):       
            colorVal = scalarMap.to_rgba(Year[i])
            colorVal = colors.rgb2hex(colorVal)
            radius = 40*np.sqrt(Lead[i])
            disc = 'Expected Lead : %i\n'%Lead[i] +\
                   'Expected Year : %i\n'%Year[i]
            folium.CircleMarker([y[i], x[i]], radius=radius,
                       popup=disc, color=None,
                       fill_color=colorVal).add_to(map)

        map.create_map(path='prediction.html')        
       
        
        

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


