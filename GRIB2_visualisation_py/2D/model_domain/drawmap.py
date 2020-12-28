# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

map = Basemap(projection='merc', lat_0=37.35, lon_0=126.58, resolution = 'l',
    urcrnrlat=44, llcrnrlat=32, llcrnrlon=121.5, urcrnrlon=132.5)
 
map.drawcoastlines()
map.drawcountries()
map.drawmapboundary()

