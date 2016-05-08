import pandas as pd
import numpy as np

data = pd.read_csv("submission_xgb.csv")

data.sort(columns=['predicted'], axis=0, ascending=False, inplace=True)
data.reset_index(drop=True, inplace=True)

print data

import matplotlib.pyplot as plt

plt.hist(data['predicted'], bins=10)
plt.xlabel('Prob(Pb > 15 ppb)')
plt.ylabel('Number of locations')
plt.show()

#print np.unique(data['PID no Dash'].shape)

import folium

map_2 = folium.Map(location=[43.0125, -83.6875], zoom_start=13)

for i in range(5000):
    
    LAT_VALUE = data.ix[i, 'Latitude']
    LON_VALUE = data.ix[i, 'Longitude']
    CIRCLE_RADIUS = data.ix[i, 'predicted'] * 100
    print CIRCLE_RADIUS
    TEXT_ON_CLICK = 'Prob(Pb > 15 ppb) = ' + str("%.2f" % data.ix[i, 'predicted'])
    folium.CircleMarker([LAT_VALUE, LON_VALUE], radius=CIRCLE_RADIUS,
                         popup=TEXT_ON_CLICK, color='#3186cc',
                         fill_color='#3186cc').add_to(map_2)
                        
map_2.create_map(path='v2_map_of_pb_over_15_top_5000.html')
