import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.cm import get_cmap
from matplotlib.colors import from_levels_and_colors
import wrf 
from netCDF4 import Dataset
from cartopy import crs
from cartopy.feature import NaturalEarthFeature, COLORS

pd.set_option('display.max_columns', 30)

## STATION INFO
dir_stn = '/data1/storage/subeen/case2011/wrfout/plot_output/' #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
fname_stn = 'METADATA_station_loc.csv' 
df_stn = pd.read_csv(dir_stn + fname_stn)
df_stn.columns = df_stn.iloc[0].values
df_stn = df_stn.drop([0]).reset_index(drop = True)

## STN SELCTION: Select station to analyze X-Z Cross-section
#-- Enter list of stns --#
lst_stn = ['속초', '인제', '북강릉', '강릉', '대관령', '동해', '삼척', '정선군', '태백', '울진', '봉화', '청송군', '영덕', '영천', '포항', '경주시', '울산', '양산시', '울릉도', '창원시', '김해시', '마산', '부산', '거제'] #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# stn_name/ stn_id / stn_lat / stn_loc
dic_stn = {}
for stn in lst_stn:
    stn_info = {}
    stn_info['stn_id'] = df_stn[df_stn['지점명'] == stn]['지점'].values[0] # stn number in string
    stn_info['stn_lat'] = float(df_stn[df_stn['지점명'] == stn]['위도'].values[0])
    stn_info['stn_lon'] = float(df_stn[df_stn['지점명'] == stn]['경도'].values[0])
    stn_info['cross_start'] = wrf.CoordPair(lat = stn_info['stn_lat'], lon = 126)
    stn_info['cross_end'] = wrf.CoordPair(lat = stn_info['stn_lat'], lon = 134)
    dic_stn[stn] = stn_info
    
# PLOT X-Z Cross section

# Time domain list
start = dt.datetime(2011, 2, 10, 18, 0, 0)
lst_dt = [(start + dt.timedelta(hours = i)) for i in range(46)]
lst_wrfout = [f"wrfout_d02_{date.strftime('%Y-%m-%d_%H:%M:%S')}.nc" for date in lst_dt]


# Iterate trough wrfout files and plot X-Z Cross section
dir_wrfout = '/data1/storage/subeen/case2011/wrfout/' #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
dir_fig = '/data1/storage/subeen/case2011/wrfout/plot_output/CloudBoundary/' #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



csv_minmax = pd.DataFrame()
for stn in dic_stn:
    print (stn)
    lst_qmax = []
    lst_qmin = []
    
    for time in lst_dt:
        print (time.strftime('%Y-%m-%d_%H:%M:%S'))
        f = dir_wrfout + f"wrfout_d02_{time.strftime('%Y-%m-%d_%H:%M:%S')}.nc"
        ncfile = Dataset(f, 'r')
        qcloud = wrf.getvar(ncfile, 'QCLOUD')
        qice = wrf.getvar(ncfile, 'QICE')
        ter = wrf.getvar(ncfile, 'ter') # terain height
        hgt = wrf.getvar(ncfile, 'z') # model height
        Q = qcloud + qice

        cross_start = dic_stn[stn]['cross_start']
        cross_end = dic_stn[stn]['cross_end']
        Q_cross = wrf.vertcross(Q, hgt, wrfin = ncfile, start_point = cross_start, end_point = cross_end, latlon = True, meta = True)
        Q_cross.attrs.update(qcloud.attrs)
        Q_cross.attrs['description'] = "Mixing ratio Q_c + Q_i"
        Q_cross.attrs['units'] = 'kg kg-1'
        Q_cross_filled = np.ma.copy(wrf.to_np(Q_cross)) # Make a copy of the cross-section data.(regular numpy array)

        # For each cross section column, find the first index with non-missing values and copy these to the missing elements below
        for i in range(Q_cross_filled.shape[-1]):
            column_vals = Q_cross_filled[:,i]
            first_index = int(np.transpose((column_vals > -1).nonzero())[0])
            Q_cross_filled[0:first_index, i] = Q_cross_filled[first_index, i]

        # Get the terrain heights along the cross section line
        ter_line = wrf.interpline(ter, wrfin = ncfile, start_point = cross_start, end_point = cross_end)

        # Get the lat/lon points
        lats, lons = wrf.latlon_coords(Q)

        # Get the cartopy projection object
        cart_proj = wrf.get_cartopy(qcloud)

        # Create the figure
        fig = plt.figure(figsize = (16, 8))
        ax_cross = plt.axes()

        Q_max = max([max(row) for row in Q_cross_filled])
        Q_min = min([min(row) for row in Q_cross_filled])
        print (f"Q max: {Q_max}")
        lst_qmax.append(Q_max)
        lst_qmin.append(Q_min)

        Q_levels = np.array([0.,0.01, 0.1, 1.0,2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 18.0, 21.0, 24.0])*(10**(-5))

        Q_rgb = np.array([[255,255, 255],
                        [1,159,244], [3,0,244],
                        [2,253,2], [1,197,1],
                        [0,142,0], [253,248,2],
                        [229,188,0], [253,149,0],
                        [253,0,0], [212,0,0],
                        [188,0,0],[248,0,253],
                        [152,84,198]], np.float32) / 255.0

        Q_map, Q_norm = from_levels_and_colors(Q_levels, Q_rgb, extend = "max")

        xs = np.arange(0, Q_cross.shape[-1], 1)
        ys = wrf.to_np(Q_cross.coords['vertical'])
        Q_contours = ax_cross.contourf(xs, ys, wrf.to_np(Q_cross_filled), levels = Q_levels, cmap = Q_map, norm = Q_norm, extend = 'max')

        ax_cross.set_ylim(0, 5000)

        cb_Q = fig.colorbar(Q_contours, ax=ax_cross)
        cb_Q.ax.tick_params(labelsize=8)

        ht_fill = ax_cross.fill_between(xs, 0, wrf.to_np(ter_line),
                                    facecolor="black")

        coord_pairs = wrf.to_np(Q_cross.coords["xy_loc"])
        x_ticks = np.arange(coord_pairs.shape[0])
        x_labels = [pair.latlon_str() for pair in wrf.to_np(coord_pairs)]


        num_ticks = 5
        thin = int((len(x_ticks) / num_ticks) + .5)
        ax_cross.set_xticks(x_ticks[::thin])
        ax_cross.set_xticklabels(x_labels[::thin], rotation=45, fontsize=8)
        plt.grid(True, color='gray', alpha=0.5, linestyle='--')

        # Set the x-axis and  y-axis labels
        # ax_cross.set_xlabel("Lat, Lo", fontsize=12)
        ax_cross.set_ylabel("Height [m]", fontsize=12)

        # Add a title
        ax_cross.set_title(f"Station No. {dic_stn[stn]['stn_id']} : $Q_c$ + $Q_i$ [$kg$ $kg^{-1}$]", {"fontsize" : 14})

        time_str = time.strftime('%Y-%m-%d_%H:%M:%S')

        path = os.path.join(dir_fig, dic_stn[stn]['stn_id'])'
        if not os.path.exists(path):
            os.mkdir(path)
        id_stn = dic_stn[stn]['stn_id']
        plt.savefig(path +'/' +  id_stn + f"CloudBoundary_{time_str}", dpi = 200)

        plt.close(fig)

    print ("qmax", lst_qmax)
    print ("qmin", lst_qmin)
    print ("stn", stn, "max", max(lst_qmax), "min", min(lst_qmin))

    csv_minmax['stn'] = stn
    csv_minmax ['qmin'] = min(lst_qmin)
    csv_minmax['qmax'] = max(lst_qmax)
    csv_minmax['lst_qmin'] = lst_qmin
    csv_minmax['lst_qmax'] = lst_qmax

    csv_minmax.to_csv('Q_minmax.csv')






