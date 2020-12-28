import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from wrf import getvar, interpline, CoordPair, xy_to_ll, ll_to_xy
from netCDF4 import Dataset
pd.set_option('display.max_columns', 30)

dir = '/data1/storage/subeen/case2011/wrfout/'
savedir = '/data1/storage/subeen/case2011/wrfout/plot_output/compare_pcp/'
obs = pd.read_csv(dir + 'plot_output/OBS_ASOS_TIM_20200701160857.csv')
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

stn = pd.read_csv(dir+ 'plot_output/METADATA_station_loc.csv')
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
    
latlon_list = []
for key in stn_obs:
    latlon_list.append((stn_obs[key]['stn_lat'], stn_obs[key]['stn_lon']))

time_list_wrf = [(start + datetime.timedelta(hours = i)).strftime('%Y-%m-%d_%H:%M:%S') for i in range(46)]
ncfile = Dataset(dir + "wrfout_d02_2011-02-12_15:00:00.nc")

for i, stn_key in enumerate(list(stn_obs.keys())):
    print (stn_obs[stn_key]['stn_num'])
    stn_lat = stn_obs[stn_key]['stn_lat']
    stn_lon = stn_obs[stn_key]['stn_lon']
    stn_i = ll_to_xy(ncfile, stn_lat, stn_lon).values[0]
    stn_j = ll_to_xy(ncfile, stn_lat, stn_lon).values[1]
    pcp_obs = list(stn_obs[stn_key]['data']['적설(cm)'].values)
    #print (pcp_obs)
    pcp_model = []
    snow_model = []
    snowh_model = []
    df_pcp_model = pd.DataFrame()
    df_snow_model = pd.DataFrame()
    for t in time_list_wrf:
        file =  Dataset(dir + f"wrfout_d02_{t}.nc")
        rainc = getvar(file, 'RAINC')[stn_i, stn_j].values
        rainnc = getvar(file, 'RAINNC')[stn_i, stn_j].values
        rain = rainc + rainnc
        snowc = getvar(file, 'SNOWC')[stn_i, stn_j].values
        snownc = getvar(file, 'SNOWNC')[stn_i, stn_j].values
        snowh = getvar(file, 'SNOWH')[stn_i, stn_j].values
        snow = snowc + snownc 
        pcp_model.append(rain)
        snow_model.append(snow)
        snowh_model.append(snowh)
    #print('pcp_model', pcp_model)
    #print('snow_model', snow_model)
    df_pcp_model[stn_key] = pcp_model
    df_snow_model[stn_key] = snow_model

    x_axis = range(46)
    plt.figure()
    plt.plot(pcp_obs, marker = '+', color ='C9', label = 'SnowHeight_obs')
    plt.plot(pcp_model, linestyle = '--', color ='C2', label = 'Rain_model')
    plt.plot(snowh_model, linestyle = ':', color ='C8', label = 'SnowHeight_model')
    plt.plot(snow_model, linestyle = '-.', color = 'C0', label = 'Snow_model')
    plt.xlabel('Hours Since 2011-02-10 18:00:00 UTC')
    plt.ylabel('RAIN/SNOW[mm]')
    plt.title(f"{stn_obs[stn_key]['stn_num']}")
    plt.xlim(0, 46)
    plt.ylim(0, 140)
    plt.legend(loc = 'upper left')
    plt.tight_layout()
    plt.savefig(f"{stn_obs[stn_key]['stn_num']}_compare_pcp.png", dpi = 300)
    
    
    
        
    
        
        








