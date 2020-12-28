import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import from_levels_and_colors
import matplotlib.ticker as ticker
import wrf 
from netCDF4 import Dataset
from cartopy import crs
from cartopy.feature import NaturalEarthFeature, COLORS
from wrf import getvar
import metpy
import units
import warnings
warnings.filterwarnings("ignore")
#import metpy_calc


pd.set_option('display.max_columns', 30)

def wind_dir(u_ms, v_ms):
    wind_abs = np.sqrt(u_ms**2 + v_ms**2)
    wind_dir_trig_to = np.arctan2(u_ms/wind_abs, v_ms/wind_abs) 
    wind_dir_trig_to_degrees = wind_dir_trig_to * 180/np.pi + 180
    return wind_dir_trig_to_degrees



## STATION INFO
dir_stn = '/data1/storage/subeen/case2011/wrfout/plot_output/'
fname_stn = 'METADATA_station_loc.csv'

dir_obs = '/data1/storage/subeen/case2011/wrfout/plot_output/'
fname_obs = 'OBS_ASOS_TIM_20200701160857.csv'

dir_wrf = '/data1/storage/subeen/case2011/wrfout/'


################### STN INFO ###################
df_stn = pd.read_csv(dir_stn + fname_stn)
df_stn.columns = df_stn.iloc[0].values
df_stn = df_stn.drop([0]).reset_index(drop = True)

################### TIME DOMAIN ###################
start = dt.datetime(2011, 2, 10, 18, 0, 0)
lst_dt = [(start + dt.timedelta(hours = i)) for i in range(46)]

################### OBS RESULT ###################
obs = pd.read_csv(dir_obs + 'OBS_ASOS_TIM_20200701160857.csv')
obs['일시'] +=':00'
utc_date = []



# 20S/FAA/FAA_Codes/u10:v10-pcp/wrfout_d02_2011-02-12_15:00:00.nc

for i in range(len(obs)):
    time = dt.datetime.strptime(obs.iloc[i]['일시'], '%Y-%m-%d %H:%M:%S') + dt.timedelta(hours = 9)
    utc_date.append(time.strftime('%Y-%m-%d_%H:%M:%S'))

obs['UTC'] = utc_date
cols = list(obs.columns)
cols[3], cols[-1] = cols[-1], cols[3]
obs = obs[cols]
obs = obs.drop(['일시'], axis = 1)

start = dt.datetime(2011, 2, 10, 18, 0, 0)
time_list = [start +dt.timedelta(hours = i*3) for i in range(16)]
time_list = [dt.strftime('%Y-%m-%d_%H:%M:%S') for dt in time_list]

for idx in range(len(obs)):
    if obs['UTC'][idx] not in time_list:
        obs = obs.drop(idx)
        
obs = obs.reset_index(drop = True)
        
obs = obs.reset_index(drop = True)

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
        
@ticker.FuncFormatter
def wind_dir(x, pos):
    if x == 0:
        return "N"
    elif x == 45:
        return "NE"
    elif x == 90:
        return "E"
    elif x == 135:
        return "SE"
    elif x == 180:
        return "S"
    elif x == 225:
        return "SW"
    elif x == 270:
        return "W"
    elif x == 315:
        return "NW"
    elif x == 360:
        return "N"
    

plt.rcParams["figure.figsize"] = (40,5)
plt.rcParams["lines.linewidth"] = 2

        
        
stn_lst = ['속초', '인제', '북강릉', '강릉', '대관령', '동해', '삼척', '정선군', '태백', '울진', '봉화', '청송군', '영덕', '영천', '포항', '경주시', '울산', '양산시', 
           '울릉도', '창원시', '김해시', '마산', '부산', '거제']

dataset = [Dataset(dir_wrf + f"wrfout_d02_{time}.nc") for time in time_list]
ncfile = dataset[0]

for stn in stn_lst:
    
    if stn in obs.지점명.values:

        infos = obs[obs['지점명'] == stn]
        stn_id = infos['지점'].values[0]
        stn_name = stn
        stn_lat = float(df_stn[df_stn['지점명'] == stn]['위도'].values[0])
        stn_lon = float(df_stn[df_stn['지점명'] == stn]['경도'].values[0])
        stn_i = wrf.ll_to_xy(ncfile, stn_lat, stn_lon).values[0]
        print (stn_i)
        stn_j = wrf.ll_to_xy(ncfile, stn_lat, stn_lon).values[1]
        print (stn_j)
        
        
        
        rainc = np.array([getvar(file, 'RAINC')[stn_i, stn_j] for file in dataset])
        rain_obs = np.array([obs[obs.UTC == time][obs.지점 == stn_id]['강수량(mm)'].values[0] for time in time_list])
        rainnc = np.array([getvar(file, 'RAINNC')[stn_i, stn_j] for file in dataset])
        rain = rainc + rainnc
        rain_obs = np.array([obs[obs.UTC == time][obs.지점 == stn_id]['강수량(mm)'].values[0] for time in time_list])

        np.nan_to_num(rain, 0)
                          
        u = np.array([getvar(file, 'U')[0, stn_i, stn_j] for file in dataset])
        v = np.array([getvar(file, 'V')[0, stn_i, stn_j] for file in dataset])
        print (np.shape(u))
        print (np.shape(v))


        wind_direction = np.array([wrf.g_uvmet.get_uvmet_wdir(file)[0, stn_i, stn_j] for file in dataset])
	
        wind_direction_obs = np.array([obs[obs.UTC == time][obs.지점 == stn_id]['풍향(16방위)'].values[0] for time in time_list])
        wind_speed = np.array([np.sqrt(u[i]**2+v[i]**2) for i in range(len(u))])
        wind_speed_obs = np.array([obs[obs.UTC == time][obs.지점 == stn_id]['풍속(m/s)'].values[0] for time in time_list])
        
        RH = np.array([getvar(file, 'rh')[0, stn_i, stn_j] for file in dataset])
        RH_obs = np.array([obs[obs.UTC == time][obs.지점 == stn_id]['습도(%)'].values[0] for time in time_list])
        
        pressure = np.array([getvar(file, 'slp')[ stn_i, stn_j] for file in dataset])
        pressure_obs = np.array([obs[obs.UTC == time][obs.지점 == stn_id]['해면기압(hPa)'].values[0] for time in time_list])
        
        temperature = np.array([getvar(file, 'tc')[0, stn_i, stn_j] for file in dataset])
        temperature_obs = np.array([obs[obs.UTC == time][obs.지점 == stn_id]['기온(°C)'].values[0] for time in time_list])
        
        snowc =  np.array([getvar(file, 'SNOWC')[ stn_i, stn_j] for file in dataset])
        snownc =  np.array([getvar(file, 'SNOWNC')[ stn_i, stn_j] for file in dataset])
        snow = snowc + snownc
        snow_obs = np.array([obs[obs.UTC == time][obs.지점 == stn_id]['적설(cm)'].values[0] for time in time_list])
                           

        # RAIN/WIND DIRECTION/WINDSPEED/TEMPERATURE // PRESSURE / HUMIDITY

        hr = np.arange(0, 46, 3)

        fig, host = plt.subplots() # Rain
        fig.subplots_adjust(right=0.75)
        host.xaxis.grid()
        host.yaxis.grid()

        par1 = host.twinx() # Right : Temperature

        par2 = host.twinx() # Left : Snowfall 
        par2.spines["left"].set_position(("axes", -0.03))
        par2.yaxis.set_label_position('left')
        par2.yaxis.set_ticks_position('left')
        make_patch_spines_invisible(par2)
        par2.spines["left"].set_visible(True)

        par3 = host.twinx() # Right : Humidity
        par3.spines["right"].set_position(("axes", 1.03))
        par3.yaxis.set_label_position('right')
        par3.yaxis.set_ticks_position('right')
        make_patch_spines_invisible(par3)
        par3.spines["right"].set_visible(True)

        par4 = host.twinx() # Right : Wind Speed
        par4.spines["right"].set_position(("axes", 1.06))
        par4.yaxis.set_label_position('right')
        par4.yaxis.set_ticks_position('right')
        make_patch_spines_invisible(par4)
        par4.spines["right"].set_visible(True)
        
        """
        par5 = host.twinx() # Right : Wind Direction
        par5.spines["right"].set_position(("axes", 1.09))
        par5.yaxis.set_label_position('right')
        par5.yaxis.set_ticks_position('right')
        par5.yaxis.set_major_formatter(wind_dir)
        make_patch_spines_invisible(par5)
        par5.spines["right"].set_visible(True)

        par6 = host.twinx() # Right : Surface Pressure
        par6.spines["right"].set_position(("axes", 1.12))
        par6.yaxis.set_label_position('right')
        par6.yaxis.set_ticks_position('right')
        make_patch_spines_invisible(par6)
        par6.spines["right"].set_visible(True)
        """


        # Offset the right spine of par2.  The ticks and label have already been
        # placed on the right by twinx above.

        # make_patch_spines_invisible(par2)


        # Having been created by twinx, par2 has its frame off, so the line of its
        # detached spine is invisible.  First, activate the frame but make the patch
        # and spines invisible.

        # Second, show the right spine.


        p1, = host.plot(hr, rain, "b", linestyle = ':', marker = 'p', fillstyle = 'none', label="Precipitation_model")
        p1_1, = host.plot(hr, rain_obs, "b", linestyle = 'None', marker = 'p', label="Precipitation_obs")        
        
        p2, = par1.plot(hr, wind_direction, "C5", linestyle = 'None', marker = 'o', fillstyle = 'none', label="Wind Direction_model")
        p2_1, = par1.plot(hr, wind_direction_obs, "C5", linestyle = 'None', marker = '8', label="Wind Direction_obs")
        
        p3, = par2.plot(hr, snow, "C9", linestyle = ':', marker = 'p', fillstyle = 'none', label="Snowfall_model")
        p3_1, = par2.plot(hr, snow_obs, "C9", linestyle = 'None', marker = 'p', label="Snowfall_obs")
        
    
        p4, = par3.plot(hr, wind_speed, "C1", linestyle = ':', marker = 'p', fillstyle = 'none', label = "Wind Speed_model")
        p4_1, = par3.plot(hr, wind_speed_obs, "C1", linestyle = 'None', marker = 'p', label="Wind Speed_obs")
        
        
        
        """
        p5, = par4.plot(hr, wind_speed, "C1", label = "Wind Speed")
        p6, = par5.plot(hr, wind_direction, linestyle = 'None', marker = 'o', color = 'C5' , label = "Wind Direction")
        p7, = par6.plot(hr, pressure, "C2", label = "pressure")"""

        host.set_xlim(0, 45)
        host.set_ylim(0, 40) # Pcp
        plt.xticks(np.arange(0, 46, step=1), [f"{x}H" for x in np.arange(0, 46, step=1)])
        # par1.set_ylim(-5, 5) # Temperature
        par1.set_ylim(0, 360) # Wind Direction
        par2.set_ylim(0, 10) #Snowfall
        par3.set_ylim(0, 20) # Wind Speed
        
        
        # par2.set_ylim(0, 10) # Snowfall
        # par3.set_ylim(0, 100) # Humidity
        # par4.set_ylim(0, 20) # Windspeed
        # par5.set_ylim(0, 360) # WindDirection
        # par6.set_ylim(980, 1030) # Pressure


        host.set_xlabel("Hours since UTC 2011-02-10 18:00:00")
        host.set_ylabel("Rain[mm]")
        par1.set_ylabel("Wind Direction[$^\circ$]")
        par2.set_ylabel("Snowfall[cm]")
        par3.set_ylabel("Wind Speed[m/s]")
        """par4.set_ylabel("Wind Speed[m/s]")
        par5.set_ylabel("Wind Direction[$^\circ$]")
        par6.set_ylabel("Pressure[hpa]")"""

        host.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p2.get_color())
        par2.yaxis.label.set_color(p3.get_color())
        par3.yaxis.label.set_color(p4.get_color())
        """par4.yaxis.label.set_color(p5.get_color())
        par5.yaxis.label.set_color(p6.get_color())
        par6.yaxis.label.set_color(p7.get_color())"""
        
        positions = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        par1.yaxis.set_major_locator(ticker.FixedLocator(positions))
        par1.yaxis.set_major_formatter(wind_dir)

        tkw = dict(size=4, width=1.5)
        host.tick_params(axis='y', colors=p1.get_color(), **tkw)
        par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
        par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
        par3.tick_params(axis='y', colors=p4.get_color(), **tkw)
        """par4.tick_params(axis='y', colors=p5.get_color(), **tkw)
        par5.tick_params(axis='y', colors=p6.get_color(), **tkw)
        par6.tick_params(axis='y', colors=p7.get_color(), **tkw)"""

        host.tick_params(axis='x', **tkw)

        lines = [p1, p1_1, p2, p2_1, p3, p3_1, p4, p4_1]

        host.legend(lines, [line.get_label() for line in lines])
        host.set_title(f"{stn_id}_MODEL")

        plt.savefig(f'./{stn_id}_{stn}_Wind+RainSnow_compare.png')
        plt.show()
        plt.close()





