import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.get_fontconfig_fonts()
from wrf import getvar, interpline, CoordPair, xy_to_ll, ll_to_xy
from netCDF4 import Dataset
import seaborn as sns
from mpl_toolkits.basemap import Basemap
pd.set_option('display.max_columns', 30)
#-*- coding:utf-8 -*-
import matplotlib

dir = '/data1/storage/subeen/case2011/wrfout/plot_output/'
obs = pd.read_csv(dir + 'OBS_ASOS_TIM_20200701160857.csv')
obs['일시'] +=':00'
utc_date = []

for i in range(len(obs)):
    dt = datetime.datetime.strptime(obs.iloc[i]['일시'], '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours = 9)
    utc_date.append(dt.strftime('%Y-%m-%d %H:%M:%S'))

obs['UTC'] = utc_date
cols = list(obs.columns)
cols[3], cols[-1] = cols[-1], cols[3]
obs = obs[cols]
obs = obs.drop(['일시'], axis = 1)

start = datetime.datetime(2011, 2, 10, 18, 0, 0)
time_list = [start + datetime.timedelta(hours = i) for i in range(46)]
time_list = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in time_list]

for idx in range(len(obs)):
    if obs['UTC'][idx] not in time_list:
        obs = obs.drop(idx)
        
obs = obs.reset_index(drop = True)
obs_dict = {}

stn = pd.read_csv(dir + 'METADATA_station_loc.csv')
stn.columns = stn.iloc[0].values
stn = stn.drop(0).reset_index(drop = True)
stn_num = list(stn['지점'].values)
stn_name = list(stn['지점명'].values)

stn_obs = {}
for loc in stn_num:
    loc_int = int(loc)
    obs_dict = {}
    obs_dict['stn_num'] = loc
    obs_dict['stn_lat'] = float(stn[stn['지점'] == loc]['위도'].values[0])
    obs_dict['stn_lon'] = float(stn[stn['지점'] == loc]['경도'].values[0])
    obs_dict['data'] = obs[obs['지점'] == loc_int]
    stn_obs[loc + f"_{stn[stn['지점'] == loc]['지점명'].values[0]}"] = obs_dict   
    


plt.figure(figsize=(20,20))

map = Basemap(projection='merc', lat_0=37.35, lon_0=126.58, resolution = 'h',
    urcrnrlat=40, llcrnrlat=32, llcrnrlon=121.5, urcrnrlon=132.5)
map.drawcoastlines()
map.drawcountries()
map.drawmapboundary()

lon = []
lat = []
labels = []
for stn_name_id in stn_obs:
    stn_label = stn_obs[stn_name_id]['stn_num']
    stn_lon = stn_obs[stn_name_id]['stn_lon']
    stn_lat = stn_obs[stn_name_id]['stn_lat']
    
    lon.append(stn_lon)
    lat.append(stn_lat)
    labels.append(stn_label)

x,y = map(lon, lat)
map.plot(x, y, 'bo', markersize=7)

for label, xpt, ypt in zip(labels, x, y):
    plt.text(xpt - 19, ypt + 0.4, label, fontsize = 14)

plt.savefig('stn_loc.png', dpi = 300)
plt.show()




