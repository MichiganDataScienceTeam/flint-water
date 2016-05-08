import numpy as np
import sys


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



def mainPipeline():

    import sys; import numpy as np
    import numpy.random as npr 
    import matplotlib.pylab as plt
    from sklearn.cross_validation import train_test_split
    import scipy.io as scio
    import pandas as pd    
    npr.seed(1000)

    print( "READY" )
    
    try: stepNum = int(sys.argv[1])
    except IndexError: sys.exit(-1)

    if stepNum == 0:
        # look into catalogs
        data = pd.read_csv('./data/lead_parcel_data_train.csv')
        #print data.columns
        #print data[:3]
        data['Lead (ppb)'] += 1.
        data['Lead (ppb)'] = data['Lead (ppb)'].apply(np.log10)
        print data['Year Built'].unique()
        #print data['Longitude'].unique()
        print data['Prop Class'].unique()

    if stepNum == 1: 

        from networkPipeline import network_pipeline

        nw = network_pipeline()
        nw.generate_network()
        nw.analyse_netwrok()
        nw.realization_netwrok()

    if stepNum == 2: 

        from densityPipeline import density_pipeline

        densp = density_pipeline()
        densp.generate_density_map(npix=100)
        densp.analyse_density_map()
        
        
    if stepNum == 3: 

        from densityPipeline import density_pipeline

        densp = density_pipeline()

        #densp.generate_density_map(npix=120)
        #densp.analyse_density_map()
        densp.animation_density_map()
    
   
    if stepNum == 4: 

        from networkPredictionPipeline import network_pipeline

        nw = network_pipeline(n_neighbour=10)
        nw.generate_network()
        nw.realize_network()
        nw.over_plot_googlemap()

        
        
        

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


