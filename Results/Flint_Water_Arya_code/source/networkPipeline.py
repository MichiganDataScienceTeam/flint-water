import numpy as np
import sys
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
import scipy.io as scio
import pandas as pd    
import scipy.stats




def regularize(x):
   x[x < 0.] = 0.
   x[x > 1.] = 1.
   return x
   #x = np.around(x)


def readFile(fname, train=True):
    import pandas as pd 
    X = pd.read_csv(fname)
    if train:
        target = X['DRUNK_DR']
        X = X.drop('DRUNK_DR',1)
    else: 
        target = np.array(X['ID'])
    X = X.drop('ID',1)
    for label in X.columns:  X[label] = X[label].replace(np.nan,-1)
    nonnumLabels = X.select_dtypes(exclude=['int','float']).axes[1]
    for label in nonnumLabels: X = X.drop(label,1)

    return X, target



class network_pipeline():


    def __init__(self, n_neighbour=10):

        from model import graph_model, graph_spatial_model

        self.n_neighbour = n_neighbour 
        self.graph = graph_model( n_neighbour=n_neighbour )
        #self.graph = graph_spatial_model( dist=0.0002 )
        self.data = pd.read_csv('./data/sentinel_test_data.csv')
        self.clean_dataset()


    def clean_dataset(self):

        for iLabel in ['Latitude','Longitude']:
            self.data[iLabel] = self.data[iLabel].fillna(self.data[iLabel].iloc[0])

        self.data = self.data[np.array(self.data['Year Built'] > 10)]
        #self.data = self.data[np.array(self.data['Lead (ppb)'] > 0.1)]
        #self.data = self.data[np.array(self.data['Copper (ppb)'] > 0.1)]
        #self.data = self.data[np.array(self.data['Use Type'] == 'Residential' )]
        print "Length data :", len(self.data) 


    def generate_network(self):

        def regulate_distance(x):
            x = (x - min(x)) / (max(x) - min(x))
            return x

        # generate network

        L1 = np.array(self.data['Latitude']) 
        L1 = np.sqrt(regulate_distance(L1)) 
        L2 = np.array(self.data['Longitude']) 
        L2 = np.sqrt(regulate_distance(L2))
        #PropClass = np.array(self.data['Prop Class'].apply(hash)%4) / 5.
        #PA = np.array((self.data['Parcel Acres']+1).apply(np.log10))
        #PA *= .01
        #print min(PA), max(PA)
        #Rental = np.array(self.data['Rental'].apply(hash)%2) / 20.
        #HydrantType = np.array( self.data['Hydrant Type'].apply(hash) / 2.)
        #year = np.array(self.data['Year Built']) / 583.0

        self.graph.generate_directed_graph( (L1,L2), verbose=False )
        self.graph.save_graph(fdir='./output/graph/model_')



    def analyse_netwrok(self): 
          
        from analysis_tool import analysis_n_neighbour_graph
        graphTool = analysis_n_neighbour_graph(self.n_neighbour)

        def print_corr(graph, y, tag='Lead level'):
            mean_n_y = \
                 graphTool.mean_neighbour_value(\
                      graph.row, graph.col, graph.wight, graph.n_nodes, y)
            mask = ~np.isnan(mean_n_y)
            print "Pearson Correlation node and its neighbour for %s:%0.3f"\
            %(tag,scipy.stats.pearsonr(y[mask], mean_n_y[mask])[0]),
            print "(p-val:%0.3f)"\
            %scipy.stats.pearsonr(y[mask], mean_n_y[mask])[1]

        def print_neighbour(graph):   
            mean_n, std_n = \
                 graphTool.mean_neighbour(\
                      graph.row, graph.col, graph.wight, graph.n_nodes)
            print " Mean and std of neighbour :", mean_n, std_n


        year  = np.array(self.data['Year Built']) / 83.0
        PropValue = np.array(self.data['Residential Building Value'].apply(np.log10))
        PropValue *= 1.

        PA = 1./np.array(self.data['Parcel Acres'].apply(np.log10))
        PA *= .01

        mymap = {'A.D.':1,'T.C.':2,'Other':3,'Dar':4,'Mueller':5}
        self.data['Hydrant Type'] = self.data['Hydrant Type'].replace(mymap)
        HydrantType = np.array( self.data['Hydrant Type'].apply(hash) / 1000. )

        Storeys   = np.array(self.data['Building Storeys']) * 4.
        PropClass = np.array(self.data['Prop Class'].apply(hash)%4) / 4.
        PropClass *= 1.
        UseType = np.array(self.data['Use Type'].apply(hash)%3) * 10.
        Cond = np.array(self.data['Housing Condition 2014'].apply(hash)%8) / 64.
        Rental = np.array(self.data['Rental'].apply(hash)%2) 

        #print data['Hydrant Type'].unique()
        #print data['Hydrant Type'].apply(hash).unique()
       
        #Lead = np.log10(np.array(self.data['Lead (ppb)']) + 0.1)

        Lead = np.array(self.data['Lead (ppb)'])
        Lead = np.log10(Lead + 1.0)

        Copper = np.array(self.data['Copper (ppb)'])
        Copper = np.log10(Copper + 1.0)

        print_neighbour(self.graph)   
        print_corr(self.graph, Lead, tag='Lead level')
        print_corr(self.graph, Copper, tag='Copper level')
        print_corr(self.graph, year, tag='Year')
        print_corr(self.graph, HydrantType, tag='Hydrant Type')
        print_corr(self.graph, PropValue, tag='Prop Value')
        print_corr(self.graph, Rental, tag='Rental')
        print_corr(self.graph, PA, tag='Parcel Acres')

        ####### 
        print "None local correlation "
        print "Year & Lead : ", scipy.stats.pearsonr(Lead, year)[0]
        print "Year & Copper : ", scipy.stats.pearsonr(Copper, year)[0]
        print "HydrantType & Lead : ", scipy.stats.pearsonr(HydrantType, year)[0]
        print "PropValue & Lead : ", scipy.stats.pearsonr(PropValue, year)[0]
        print "Copper & Lead : ", scipy.stats.pearsonr(PropValue, year)[0]


    def realization_netwrok(self): 

        from analysis_tool import analysis_n_neighbour_graph
        graphTool = analysis_n_neighbour_graph(self.n_neighbour)

        def make_xy_plot(graph, tag_1, tag_2):

            import matplotlib.pyplot as plt
            import matplotlib.colors as colors
            import matplotlib.cm as cmx

            Lead = np.log10( np.array(self.data['Lead (ppb)']) + 1.0 )
            plt.figure(figsize=(20,20))
            jet    = plt.get_cmap('seismic')
            cNorm  = colors.Normalize(vmin=0.3, vmax=1.3)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

            for x, y, col in zip(np.array(self.data[tag_1]), 
                            np.array(self.data[tag_2]), Lead):
                colorVal = scalarMap.to_rgba(col)
                plt.plot(x, y, '.', color=colorVal)

            plt.savefig('./output/graph_vis/Lead.png', bbox_inches='tight')


        def make_graph_node_correlation_plot(graph, val):

            plt.figure(figsize=(10,10))

            for irow, icol in zip(graph.row, graph.col):
                plt.plot(val[irow], val[icol], '.', color='black')

            plt.xlabel('Node value')
            plt.ylabel('Neighbour value')

            #plt.savefig('xxx.png', bbox_inches='tight')


        def make_graph_mean_node_correlation_plot(graph, val):

            mean_n_val = \
                 graphTool.mean_neighbour_value(\
                      graph.row, graph.col, graph.wight, graph.n_nodes, val)
            mask = ~np.isnan(mean_n_val)

            plt.figure(figsize=(10,10))

            plt.plot(val[mask], mean_n_val[mask], '.', color='blue', alpha=0.1)

            plt.xlabel('Node value')
            plt.ylabel('Mean Neighbour value')

            plt.savefig('xxx.png', bbox_inches='tight')




        #make_xy_plot(self.graph, 'Longitude', 'Latitude')
        #make_graph_node_correlation_plot(self.graph, 'Lead (ppb)')
        #make_graph_node_correlation_plot(self.graph, 'Residential Building Value')
        #make_graph_node_correlation_plot(self.graph, 'Year Built')

        #val = np.log10( np.array(self.data['Lead (ppb)']) + 1.0 )
        #val = np.log10( np.array(self.data['Residential Building Value']) + 1.0 )
        #val = np.array(self.data['Year Built']) 
        #make_graph_mean_node_correlation_plot(self.graph, val)
        
        
        



        
        
        

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


