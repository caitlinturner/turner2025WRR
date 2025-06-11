# -*- coding: utf-8 -*-
"""
@author: caitlinturner
"""

import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, num2date
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
from dataretrieval import nwis
from noaa_coops import Station
import seaborn as sns
from goodness_of_fit import d, r_pearson, rmse
from matplotlib.dates import DateFormatter, MonthLocator
from tabulate import tabulate
from datetime import datetime
from dateutil.relativedelta import relativedelta
import cmocean
import warnings
warnings.filterwarnings("ignore")

## Import Model Data ## -----------------------------------------------------
sd = '2014-01-01'
ed = '2014-12-31'
sdm = '2021-01-01 00:00:00'
edm = '2021-12-31 23:00:00'
lw = 1.25
 
# Upload Data
data_xr_his = xr.open_dataset('Calibration/FlowFM_his.nc')
station_names = data_xr_his.station_name.astype(str).to_pandas()
df = data_xr_his.waterlevel.to_pandas()
df.columns = station_names
df.index = df.index - pd.DateOffset(hours=0)



## Set Up Figure and Stats Table ## ----------------------------------------
# Figure
ytickmin, ytickmax, ytickstep = -0.5,2,3
ymin,ymax = -0.5, 2.6
xstattxt, ystattxt = 0.315, 0.8
xtitle, ytitle =  0.01, 0.6

xx = 0.04  
yy= 0.48

fig, axs = plt.subplots(nrows=6, ncols=2, sharex=True, sharey=True, figsize=(6.4, 3.6), dpi = 600)  
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
plt.subplots_adjust(hspace=0.3,wspace = 0.05)
sns.set_palette("colorblind")
colors = [cmocean.cm.thermal_r(0.9),cmocean.cm.thermal_r(0.5)] #cmocean.cm.thermal_r(0.1), 
linestyles_used = ['-','--']


fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.ylabel('Water level (m above MLLW)', labelpad=8, fontsize=7)
 
# Stats Table
stats_table = pd.DataFrame(columns=['Location', 'Index of Agreement', 'Pearson Corr Coeff', 'Root Mean Squared Error'])

# Set Moving Average to Remove Small Wind Variations (NOAA PROTOCOL, 4 Hours)
move_mean = 2

begin = datetime(2021, 1, 1, 0, 0)
dates = []
for i in range(8760):
    dates.append(begin)
    begin += relativedelta(hours=1)
dates = pd.DataFrame({'date': dates,'dateplt': dates})
dates['date'] = dates['date'].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))



# ----------------------- ### CALIBRATION ### ------------------------------ #

## NOAA GAUGES  ## ---------------------------------------------------------
# New Canal Station NOAA
df_noaa_newcanal = Station(id="8761927")
df_noaa_newcanal = df_noaa_newcanal.get_data(begin_date="20131130",
                                              end_date="20150101",
                                              product="hourly_height",
                                              datum="MLLW",
                                              units="metric",
                                              time_zone="gmt")
obs = df_noaa_newcanal.v[sd:ed].reset_index(drop=True)
model = df.NewCanal_USGS[sdm:edm].reset_index(drop=True)
datum_diff = (model - obs).mean()
model = model - datum_diff
dffig = pd.DataFrame({'date': dates.date,'Model': model, 'Obs': obs})
dffig['date'] = pd.to_datetime(dffig['date'], format="%d/%m/%Y %H:%M")
dffig.set_index('date', inplace=True)
dffig = dffig.resample('D').mean()
dffig = dffig.rolling(move_mean, min_periods=1).mean()
obs = dffig['Obs']
model = dffig['Model']
stats_table.loc[len(stats_table.index)] = ['New Canal (NOAA: 8761927)', round(d(obs, model), 2),
                                            round(r_pearson(obs, model), 2), round(rmse(obs, model),2)]

series_labels = ['Model', 'Obs'] 
for i, series in enumerate(series_labels):
    axs[0,0].plot(dffig.index, dffig[series], linewidth=lw, color=colors[i], label=series, linestyle = linestyles_used[i])
axs[0,0].set_ylim([ymin, ymax])
axs[0,0].tick_params(axis='y', which='major', labelsize=7)
axs[0,0].set_title('a.', fontsize=7, fontweight='bold', loc='center', x = xx, y = yy)
axs[0,0].text(xstattxt, ystattxt, 
            f'$d$ = {stats_table.iloc[0][1]:.2f}, '  
            f'r$^2$ = {stats_table.iloc[0][2]:.2f}, ' 
            f'RMSE = {stats_table.iloc[0][3]:.2f}',  
            transform=axs[0,0].transAxes,  
            fontsize=7,
            verticalalignment='center')
axs[0,0].set_yticks(np.linspace(ytickmin, ytickmax, ytickstep))
axs[0,0].yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}')) 
axs[0,0].text(0.4, 1.15, 
            'Calibration',  
            transform=axs[0,0].transAxes,  
            fontsize=7,
            verticalalignment='center')
axs[0,0].text(xx + 0.025, yy+0.225, 'New Canal',  transform=axs[0,0].transAxes,  fontsize=7)

# Shell Beach Station NOAA
df_noaa_shellbeach = Station(id="8761305")
df_noaa_shellbeach = df_noaa_shellbeach.get_data(begin_date="20131130",
                                              end_date="20150101",
                                                  product="hourly_height",
                                                  datum="MLLW",
                                                  units="metric",
                                                  time_zone="gmt")
model = df.ShellBeach_NOAA[sdm:edm].reset_index(drop=True)
obs = df_noaa_shellbeach.v[sd:ed].reset_index(drop=True)
datum_diff = (model - obs).mean()
model = model - datum_diff
dffig = pd.DataFrame({'date': dates.date,'Model': model, 'Obs': obs})
dffig['date'] = pd.to_datetime(dffig['date'], format="%d/%m/%Y %H:%M")
dffig.set_index('date', inplace=True)
dffig = dffig.resample('D').mean()
dffig = dffig.rolling(move_mean, min_periods=1).mean()
obs = dffig['Obs']
model = dffig['Model']
stats_table.loc[len(stats_table.index)] = ['Shell Beach \n (NOAA: 8761305)', round(d(obs, model), 2),
                                            round(r_pearson(obs, model), 2), round(rmse(obs, model),2)]


series_labels = ['Model', 'Obs'] 
for i, series in enumerate(series_labels):
    axs[1,0].plot(dffig.index, dffig[series], linewidth=lw, color=colors[i], label=series, linestyle = linestyles_used[i])
axs[1,0].set_ylim([ymin, ymax])
axs[1,0].tick_params(axis='y', which='major', labelsize=7)
axs[1,0].set_title('c.', fontsize=7, fontweight='bold', loc='center', x = xx, y = yy)
axs[1,0].text(xstattxt, ystattxt, 
            f'$d$ = {stats_table.iloc[1][1]:.2f}, '  
            f'r$^2$ = {stats_table.iloc[1][2]:.2f}, ' 
            f'RMSE = {stats_table.iloc[1][3]:.2f}',  
            transform=axs[1,0].transAxes,  
            fontsize=7,
            verticalalignment='center')
axs[1,0].set_yticks(np.linspace(ytickmin, ytickmax, ytickstep))
axs[1,0].yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}')) 
axs[1,0].text(xx + 0.025, yy+0.225, 'Shell Beach',  transform=axs[1,0].transAxes,  fontsize=7)

## USGS GAUGES ## ----------------------------------------------------------
parameterCode='00065'

# Rigolets USGS
site = '301001089442600'
df_usgs_rigolets = nwis.get_record(sites=site, service='iv', parameterCd=parameterCode,
                                                  start='2013-11-30', end='2015-01-01')
df_usgs_rigolets = df_usgs_rigolets.rename(
    columns={'00065': 'waterlevel', '00065_cd': 'cd', 'site_no': 'site'})

model = df.Rigolets_CRRT[sdm:edm].reset_index(drop=True)
waterlevel_df = df_usgs_rigolets.waterlevel.resample('H').mean()
waterlevel_df= waterlevel_df[sd:ed].multiply(0.3048)
obs = waterlevel_df.tz_localize(tz=None).reset_index(drop=True)
datum_diff = (model - obs).mean()
model = model - datum_diff
dffig = pd.DataFrame({'date': dates.date,'Model': model, 'Obs': obs})
dffig['date'] = pd.to_datetime(dffig['date'], format="%d/%m/%Y %H:%M")
dffig.set_index('date', inplace=True)
dffig = dffig.resample('D').mean()
dffig = dffig.rolling(move_mean, min_periods=1).mean()
obs = dffig['Obs']
model = dffig['Model']

obs_clean = obs.dropna()
model_clean = model[obs_clean.index]
stats_table.loc[len(stats_table.index)] = [f'The Rigolets \n (USGS: {site})', round(d(obs_clean, model_clean), 2),
                                            round(r_pearson(obs_clean, model_clean), 2), round(rmse(obs_clean, model_clean),2)]

series_labels = ['Model', 'Obs'] 
for i, series in enumerate(series_labels):
    axs[2,0].plot(dffig.index, dffig[series], linewidth=lw, color=colors[i], label=series, linestyle = linestyles_used[i])
axs[2,0].set_ylim([ymin, ymax])
axs[2,0].tick_params(axis='y', which='major', labelsize=7)
axs[2,0].set_title('e.', fontsize=7, fontweight='bold', loc='center', x = xx, y = yy)
axs[2,0].text(xstattxt, ystattxt, 
            f'$d$ = {stats_table.iloc[2][1]:.2f}, '  
            f'r$^2$ = {stats_table.iloc[2][2]:.2f}, ' 
            f'RMSE = {stats_table.iloc[2][3]:.2f}',  
            transform=axs[2,0].transAxes,  
            fontsize=7,
            verticalalignment='center')
axs[2,0].set_yticks(np.linspace(ytickmin, ytickmax, ytickstep))
axs[2,0].yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}')) 
axs[2,0].text(xx + 0.025, yy+0.225, 'Rigolets',  transform=axs[2,0].transAxes,  fontsize=7)



## USACE Gauges ## ----------------------------------------------------------- 
# (https://rivergages.mvr.usace.army.mil/WaterControl/new/layout.cfm)
# Site Locations (Not as simple to obtain as NOAA and USGS)
USACEinfo = pd.DataFrame({'Location': ['Mandeville', 'Pass Manchac', 'Chef Menteur'],
                      'Longitude': [-90.09228880, -90.40027778, -89.80102770],
                      'Latitude': [30.36579720, 30.28138889, 30.06673880],
                      'NAVD88Corr': [2009.55,2004.65,2004.65]})

# Upload Data
begin = datetime(2021, 1, 1, 8, 0)
dates = []
for i in range(365):
    dates.append(begin)
    begin += relativedelta(days=1)
dates = pd.DataFrame({'date': dates,'dateplt': dates})
dates['date'] = dates['date'].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
    
    

USACEdata = pd.read_excel(r'C:/Users/cturn/Documents/LP_WRR_Paper/Model_Calibration/Observation_Data/USACE_Sites_2014.xlsx')
USACEdata = pd.DataFrame({'ChefMenteur': pd.to_numeric(USACEdata['ChefMenteur'])*0.3048-0,
                      'Mandeville': pd.to_numeric(USACEdata['Mandeville'])*0.3048-0,
                      'PassManchac': pd.to_numeric(USACEdata['PassManchac'])*0.3048-0})

move_mean = 2
time_zone_shift_hour = 13

# Chef Menteur 
model = df.ChefMent_USACE[sdm:edm]
model = model[model.index.hour == time_zone_shift_hour]
model = model.reset_index(drop=True)
obs = USACEdata.ChefMenteur
datum_diff = (model - obs).mean()
model = model - datum_diff  
dffig = pd.DataFrame({'date': dates.date,'Model': model, 'Obs': obs})
dffig['date'] = pd.to_datetime(dffig['date'], format="%d/%m/%Y %H:%M")
dffig.set_index('date', inplace=True)
dffig = dffig.rolling(move_mean, min_periods=1).mean()
obs = dffig['Obs']
model = dffig['Model']
obs_clean = obs.dropna()
model_clean = model[obs_clean.index]
stats_table.loc[len(stats_table.index)] = ['Chef Menteur \n (USACE: 85750)', round(d(obs, model), 2),
                                            round(r_pearson(obs, model), 2), round(rmse(obs, model),2)]

series_labels = ['Model', 'Obs'] 
for i, series in enumerate(series_labels):
    axs[3,0].plot(dffig.index, dffig[series], linewidth=lw, color=colors[i], label=series, linestyle = linestyles_used[i])
axs[3,0].set_ylim([ymin, ymax])
axs[3,0].tick_params(axis='y', which='major', labelsize=7)
axs[3,0].set_title('g.', fontsize=7, fontweight='bold', loc='center', x = xx, y = yy)
axs[3,0].text(xstattxt, ystattxt, 
            f'$d$ = {stats_table.iloc[3][1]:.2f}, '  
            f'r$^2$ = {stats_table.iloc[3][2]:.2f}, ' 
            f'RMSE = {stats_table.iloc[3][3]:.2f}',  
            transform=axs[3,0].transAxes,  
            fontsize=7,
            verticalalignment='center')
axs[3,0].set_yticks(np.linspace(ytickmin, ytickmax, ytickstep))
axs[3,0].yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}')) 
axs[3,0].text(xx + 0.025, yy+0.225, 'Menteur',  transform=axs[3,0].transAxes,  fontsize=7)

# Mandeville 
model = df.Madev_USACE[sdm:edm]
model = model[model.index.hour == time_zone_shift_hour]
model = model.reset_index(drop=True)
obs = USACEdata.Mandeville
datum_diff = (model - obs).mean()
model = model - datum_diff  
dffig = pd.DataFrame({'date': dates.date,'Model': model, 'Obs': obs})
dffig['date'] = pd.to_datetime(dffig['date'], format="%d/%m/%Y %H:%M")
dffig.set_index('date', inplace=True)
dffig = dffig.rolling(move_mean, min_periods=1).mean()
obs = dffig['Obs']
model = dffig['Model']
stats_table.loc[len(stats_table.index)] = ['Mandeville \n (USACE: 85575)', round(d(obs, model), 2),
                                            round(r_pearson(obs, model), 2), round(rmse(obs, model),2)]

series_labels = ['Model', 'Obs'] 
for i, series in enumerate(series_labels):
    axs[4,0].plot(dffig.index, dffig[series], linewidth=lw, color=colors[i], label=series, linestyle = linestyles_used[i])
axs[4,0].set_ylim([ymin, ymax])
axs[4,0].tick_params(axis='y', which='major', labelsize=7)
axs[4,0].set_title('i.', fontsize=7, fontweight='bold', loc='center', x = xx, y = yy)
axs[4,0].text(xstattxt, ystattxt, 
            f'$d$ = {stats_table.iloc[4][1]:.2f}, '  
            f'r$^2$ = {stats_table.iloc[4][2]:.2f}, ' 
            f'RMSE = {stats_table.iloc[4][3]:.2f}',  
            transform=axs[4,0].transAxes,  
            fontsize=7,
            verticalalignment='center')
axs[4,0].set_yticks(np.linspace(ytickmin, ytickmax, ytickstep))
axs[4,0].yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}')) 
axs[4,0].text(xx + 0.025, yy+0.225, 'Mandeville',  transform=axs[4,0].transAxes,  fontsize=7)

# Pass Manchac
model = df.PassManac_USACE[sdm:edm]
model = model[model.index.hour == time_zone_shift_hour]
model = model.reset_index(drop=True)
obs = USACEdata.PassManchac
datum_diff = (model - obs).mean()
model = model - datum_diff
dffig = pd.DataFrame({'date': dates.date,'Model': model, 'Obs': obs})
dffig['date'] = pd.to_datetime(dffig['date'], format="%d/%m/%Y %H:%M")
dffig.set_index('date', inplace=True)
dffig = dffig.rolling(move_mean, min_periods=1).mean()
obs = dffig['Obs']
model = dffig['Model']
stats_table.loc[len(stats_table.index)] = ['Pass Manchac \n (USACE: 85420)', round(d(obs, model), 2),
                                            round(r_pearson(obs, model), 2), round(rmse(obs, model),2)]

series_labels = ['Model', 'Obs'] 
for i, series in enumerate(series_labels):
    axs[5,0].plot(dffig.index, dffig[series], linewidth=lw, color=colors[i], label=series, linestyle = linestyles_used[i])
axs[5,0].set_title('k.', fontsize=7, fontweight='bold', loc='center', x = xx, y = yy)
axs[5,0].text(xstattxt, ystattxt, 
            f'$d$ = {stats_table.iloc[5][1]:.2f}, '  
            f'r$^2$ = {stats_table.iloc[5][2]:.2f}, ' 
            f'RMSE = {stats_table.iloc[5][3]:.2f}',  
            transform=axs[5,0].transAxes,  
            fontsize=7,
            verticalalignment='center')

axs[5,0].tick_params(axis='y', which='major', labelsize=7)
axs[5,0].set_ylim([ymin, ymax])
axs[5,0].set_yticks(np.linspace(ytickmin, ytickmax, ytickstep))
axs[5,0].yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}')) 
axs[5,0].text(xx + 0.025, yy+0.225, 'Manchac',  transform=axs[5,0].transAxes,  fontsize=7)



# X-Axis
axs[5,0].xaxis.set_major_locator(MonthLocator())
def uppercase_month(x, pos):
    return num2date(x).strftime('%b')[0].upper()
axs[5,0].xaxis.set_major_formatter(FuncFormatter(uppercase_month))
axs[5,0].tick_params(axis='x', which='major', labelsize=7)
axs[5,0].set_xlim([dffig.index.min() - pd.Timedelta(days=2), dffig.index.max() + pd.Timedelta(days=0)])





## Import Validation Model Data ## -----------------------------------------------------
sd = '2011-01-01'
ed = '2011-12-31'
sdm = '2021-01-01 00:00:00'
edm = '2021-12-31 23:00:00'
lw = 1.25

# Upload Data
data_xr_his = xr.open_dataset('Validation/FlowFM_his.nc')
station_names = data_xr_his.station_name.astype(str).to_pandas()
df = data_xr_his.waterlevel.to_pandas()
df.columns = station_names
df.index = df.index - pd.DateOffset(hours=0) 


# ----------------------- ### VALIDATION ### ------------------------------ 
# Set Moving Average to Remove Small Wind Variations (NOAA PROTOCOL, 4 Hours)
move_mean = 2

begin = datetime(2021, 1, 1, 0, 0)
dates = []
for i in range(8760):
    dates.append(begin)
    begin += relativedelta(hours=1)
dates = pd.DataFrame({'date': dates,'dateplt': dates})
dates['date'] = dates['date'].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))


## NOAA GAUGES  ## ---------------------------------------------------------
# New Canal Station NOAA
df_noaa_newcanal = Station(id="8761927")
df_noaa_newcanal = df_noaa_newcanal.get_data(begin_date="20101130",
                                             end_date="20120101",
                                             product="hourly_height",
                                             datum="MLLW",
                                             units="metric",
                                             time_zone="gmt")
obs = df_noaa_newcanal.v[sd:ed].reset_index(drop=True)
model = df.NewCanal_USGS[sdm:edm].reset_index(drop=True)
datum_diff = 0.186 # From Calibration
model = model - datum_diff
dffig = pd.DataFrame({'date': dates.date,'Model': model, 'Obs': obs})
dffig['date'] = pd.to_datetime(dffig['date'], format="%d/%m/%Y %H:%M")
dffig.set_index('date', inplace=True)
dffig = dffig.resample('D').mean()
dffig = dffig.rolling(move_mean, min_periods=1).mean()
obs = dffig['Obs']
model = dffig['Model']
stats_table.loc[len(stats_table.index)] = ['New Canal (NOAA: 8761927)', round(d(obs, model), 2),
                                            round(r_pearson(obs, model), 2), round(rmse(obs, model),2)]

series_labels = ['Model', 'Obs'] 
for i, series in enumerate(series_labels):
    axs[0,1].plot(dffig.index, dffig[series], linewidth=lw, color=colors[i], label=series, linestyle = linestyles_used[i])
axs[0,1].set_ylim([ymin, ymax])
axs[0,1].tick_params(axis='y', which='major', labelsize=7)
axs[0,1].set_title('b.', fontsize=7, fontweight='bold', loc='center', x = xx, y = yy)
axs[0,1].text(xstattxt, ystattxt, 
            f'$d$ = {stats_table.iloc[6][1]:.2f}, '  
            f'r$^2$ = {stats_table.iloc[6][2]:.2f}, ' 
            f'RMSE = {stats_table.iloc[6][3]:.2f}',  
            transform=axs[0,1].transAxes,  
            fontsize=7,
            verticalalignment='center')
axs[0,1].set_yticks(np.linspace(ytickmin, ytickmax, ytickstep))
axs[0,1].yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}')) 
axs[0,1].text(0.4, 1.15, 
            'Validation',  
            transform=axs[0,1].transAxes,  
            fontsize=7,
            verticalalignment='center')


# Shell Beach Station NOAA
df_noaa_shellbeach = Station(id="8761305")
df_noaa_shellbeach = df_noaa_shellbeach.get_data(begin_date="20101130",
                                             end_date="20120101",
                                                 product="hourly_height",
                                                 datum="MLLW",
                                                 units="metric",
                                                 time_zone="gmt")
model = df.ShellBeach_NOAA[sdm:edm].reset_index(drop=True)
obs = df_noaa_shellbeach.v[sd:ed].reset_index(drop=True)
datum_diff = 0.0167 # From Calibration
model = model - datum_diff
dffig = pd.DataFrame({'date': dates.date,'Model': model, 'Obs': obs})
dffig['date'] = pd.to_datetime(dffig['date'], format="%d/%m/%Y %H:%M")
dffig.set_index('date', inplace=True)
dffig = dffig.resample('D').mean()
dffig = dffig.rolling(move_mean, min_periods=1).mean()
obs = dffig['Obs']
model = dffig['Model']
stats_table.loc[len(stats_table.index)] = ['Shell Beach (NOAA: 8761305)', round(d(obs, model), 2),
                                            round(r_pearson(obs, model), 2), round(rmse(obs, model),2)]


series_labels = ['Model', 'Obs'] 
for i, series in enumerate(series_labels):
    axs[1,1].plot(dffig.index, dffig[series], linewidth=lw, color=colors[i], label=series, linestyle = linestyles_used[i])
axs[1,1].set_ylim([ymin, ymax])
axs[1,1].tick_params(axis='y', which='major', labelsize=7)
axs[1,1].set_title('d.', fontsize=7, fontweight='bold', loc='center', x = xx, y = yy)
axs[1,1].text(xstattxt, ystattxt, 
            f'$d$ = {stats_table.iloc[7][1]:.2f}, '  
            f'r$^2$ = {stats_table.iloc[7][2]:.2f}, ' 
            f'RMSE = {stats_table.iloc[7][3]:.2f}',  
            transform=axs[1,1].transAxes,  
            fontsize=7,
            verticalalignment='center')
axs[1,1].set_yticks(np.linspace(ytickmin, ytickmax, ytickstep))
axs[1,1].yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}')) 

## USGS GAUGES ## ----------------------------------------------------------
parameterCode='00065'

# Rigolets USGS
site = '301001089442600'
df_usgs_rigolets = nwis.get_record(sites=site, service='iv', parameterCd=parameterCode,
                                                  start='2010-11-30', end ='2012-01-01')
df_usgs_rigolets = df_usgs_rigolets.rename(
    columns={'00065': 'waterlevel', '00065_cd': 'cd', 'site_no': 'site'})

model = df.Rigolets_CRRT[sdm:edm].reset_index(drop=True)
waterlevel_df = df_usgs_rigolets.waterlevel.resample('H').mean()
waterlevel_df= waterlevel_df[sd:ed].multiply(0.3048)
obs = waterlevel_df.tz_localize(tz=None).reset_index(drop=True)
datum_diff = -0.102 # From Calibration
model = model - datum_diff
dffig = pd.DataFrame({'date': dates.date,'Model': model, 'Obs': obs})
dffig['date'] = pd.to_datetime(dffig['date'], format="%d/%m/%Y %H:%M")
dffig.set_index('date', inplace=True)
dffig = dffig.resample('D').mean()
dffig = dffig.rolling(move_mean, min_periods=1).mean()
obs = dffig['Obs']
model = dffig['Model']
stats_table.loc[len(stats_table.index)] = [f'The Rigolets (USGS: {site})', round(d(obs, model), 2),
                                            round(r_pearson(obs, model), 2), round(rmse(obs, model),2)]

series_labels = ['Model', 'Obs'] 
for i, series in enumerate(series_labels):
    axs[2,1].plot(dffig.index, dffig[series], linewidth=lw, color=colors[i], label=series, linestyle = linestyles_used[i])
axs[2,1].set_ylim([ymin, ymax])
axs[2,1].tick_params(axis='y', which='major', labelsize=7)
axs[2,1].set_title('f.', fontsize=7, fontweight='bold', loc='center', x = xx, y = yy)
axs[2,1].text(xstattxt, ystattxt, 
            f'$d$ = {stats_table.iloc[8][1]:.2f}, '  
            f'r$^2$ = {stats_table.iloc[8][2]:.2f}, ' 
            f'RMSE = {stats_table.iloc[8][3]:.2f}',  
            transform=axs[2,1].transAxes,  
            fontsize=7,
            verticalalignment='center')
axs[2,1].set_yticks(np.linspace(ytickmin, ytickmax, ytickstep))
axs[2,1].yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}')) 


## USACE Gauges ## ----------------------------------------------------------- 
# (https://rivergages.mvr.usace.army.mil/WaterControl/new/layout.cfm)
# Site Locations (Not as simple to obtain as NOAA and USGS)
USACEinfo = pd.DataFrame({'Location': ['Mandeville', 'Pass Manchac', 'Chef Menteur'],
                      'Longitude': [-90.09228880, -90.40027778, -89.80102770],
                      'Latitude': [30.36579720, 30.28138889, 30.06673880],
                      'NAVD88Corr': [2009.55,2004.65,2004.65]})

# Upload Data
begin = datetime(2021, 1, 1, 8, 0)
dates = []
for i in range(365):
    dates.append(begin)
    begin += relativedelta(days=1)
dates = pd.DataFrame({'date': dates,'dateplt': dates})
dates['date'] = dates['date'].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
    
    

USACEdata = pd.read_excel('C:/Users/cturn/Documents/LP_WRR_Paper/Model_Validation/Observation_Data/USACE_Sites_2011.xlsx')
USACEdata = pd.DataFrame({'ChefMenteur': pd.to_numeric(USACEdata['ChefMenteur'])*0.3048-0,
                      'Mandeville': pd.to_numeric(USACEdata['Mandeville'])*0.3048-0,
                      'PassManchac': pd.to_numeric(USACEdata['PassManchac'])*0.3048-0})

move_mean = 2
time_zone_shift_hour = 13

# Chef Menteur 
model = df.ChefMent_USACE[sdm:edm]
model = model[model.index.hour == time_zone_shift_hour]
model = model.reset_index(drop=True)
obs = USACEdata.ChefMenteur
datum_diff = 0.202 # From Calibrationf 
model = model - datum_diff  
dffig = pd.DataFrame({'date': dates.date,'Model': model, 'Obs': obs})
dffig['date'] = pd.to_datetime(dffig['date'], format="%d/%m/%Y %H:%M")
dffig.set_index('date', inplace=True)
dffig = dffig.rolling(move_mean, min_periods=1).mean()
obs = dffig['Obs']
model = dffig['Model']
obs_clean = obs.dropna()
model_clean = model[obs_clean.index]
stats_table.loc[len(stats_table.index)] = ['Chef Menteur (USACE: 85750)', round(d(obs_clean, model_clean), 2),
                                            round(r_pearson(obs_clean, model_clean), 2), round(rmse(obs_clean, model_clean),2)]

series_labels = ['Model', 'Obs'] 
for i, series in enumerate(series_labels):
    axs[3,1].plot(dffig.index, dffig[series], linewidth=lw, color=colors[i], label=series, linestyle = linestyles_used[i])
axs[3,1].set_ylim([ymin, ymax])
axs[3,1].tick_params(axis='y', which='major', labelsize=7)
axs[3,1].set_title('h.', fontsize=7, fontweight='bold', loc='center', x = xx, y = yy)
axs[3,1].text(xstattxt, ystattxt, 
            f'$d$ = {stats_table.iloc[9][1]:.2f}, '  
            f'r$^2$ = {stats_table.iloc[9][2]:.2f}, ' 
            f'RMSE = {stats_table.iloc[9][3]:.2f}',  
            transform=axs[3,1].transAxes,  
            fontsize=7,
            verticalalignment='center')
axs[3,1].set_yticks(np.linspace(ytickmin, ytickmax, ytickstep))
axs[3,1].yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}')) 

# Mandeville 
model = df.Madev_USACE[sdm:edm]
model = model[model.index.hour == time_zone_shift_hour]
model = model.reset_index(drop=True)
obs = USACEdata.Mandeville
datum_diff = 0.159 # From Calibration
model = model - datum_diff  
dffig = pd.DataFrame({'date': dates.date,'Model': model, 'Obs': obs})
dffig['date'] = pd.to_datetime(dffig['date'], format="%d/%m/%Y %H:%M")
dffig.set_index('date', inplace=True)
dffig = dffig.rolling(move_mean, min_periods=1).mean()
obs = dffig['Obs']
model = dffig['Model']
obs_clean = obs.dropna()
model_clean = model[obs_clean.index]
stats_table.loc[len(stats_table.index)] = ['Mandeville (USACE: 85575)', round(d(obs_clean, model_clean), 2),
                                            round(r_pearson(obs_clean, model_clean), 2), round(rmse(obs_clean, model_clean),2)]

series_labels = ['Model', 'Obs'] 
for i, series in enumerate(series_labels):
    axs[4,1].plot(dffig.index, dffig[series], linewidth=lw, color=colors[i], label=series, linestyle = linestyles_used[i])
axs[4,1].set_ylim([ymin, ymax])
axs[4,1].tick_params(axis='y', which='major', labelsize=7)
axs[4,1].set_title('j.', fontsize=7, fontweight='bold', loc='center', x = xx, y = yy)
axs[4,1].text(xstattxt, ystattxt, 
            f'$d$ = {stats_table.iloc[10][1]:.2f}, '  
            f'r$^2$ = {stats_table.iloc[10][2]:.2f}, ' 
            f'RMSE = {stats_table.iloc[10][3]:.2f}',  
            transform=axs[4,1].transAxes,  
            fontsize=7,
            verticalalignment='center')
axs[4,1].set_yticks(np.linspace(ytickmin, ytickmax, ytickstep))
axs[4,1].yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}')) 

# Pass Manchac
model = df.PassManac_USACE[sdm:edm]
model = model[model.index.hour == time_zone_shift_hour]
model = model.reset_index(drop=True)
obs = USACEdata.PassManchac
datum_diff = 0.11 # From Calibration
model = model - datum_diff
dffig = pd.DataFrame({'date': dates.date,'Model': model, 'Obs': obs})
dffig['date'] = pd.to_datetime(dffig['date'], format="%d/%m/%Y %H:%M")
dffig.set_index('date', inplace=True)
dffig = dffig.rolling(move_mean, min_periods=1).mean()
obs = dffig['Obs']
model = dffig['Model']
obs_clean = obs.dropna()
model_clean = model[obs_clean.index]
stats_table.loc[len(stats_table.index)] = ['Pass Manchac (USACE: 85420)', round(d(obs_clean, model_clean), 2),
                                            round(r_pearson(obs_clean, model_clean), 2), round(rmse(obs_clean, model_clean),2)]

series_labels = ['Model', 'Obs'] 
for i, series in enumerate(series_labels):
    axs[5,1].plot(dffig.index, dffig[series], linewidth=lw, color=colors[i], label=series, linestyle = linestyles_used[i])
axs[5,1].set_title('l.', fontsize=7, fontweight='bold', loc='center', x = xx, y = yy)
axs[5,1].text(xstattxt, ystattxt, 
            f'$d$ = {stats_table.iloc[11][1]:.2f}, '  
            f'r$^2$ = {stats_table.iloc[11][2]:.2f}, ' 
            f'RMSE = {stats_table.iloc[11][3]:.2f}',  
            transform=axs[5,1].transAxes,  
            fontsize=7,
            verticalalignment='center')

axs[5,1].tick_params(axis='y', which='major', labelsize=7)
axs[5,1].set_ylim([ymin, ymax])

# X-Axis
axs[5,1].xaxis.set_major_locator(MonthLocator())
def uppercase_month(x, pos):
    return num2date(x).strftime('%b')[0].upper()
axs[5,1].xaxis.set_major_formatter(FuncFormatter(uppercase_month))
axs[5,1].tick_params(axis='x', which='major', labelsize=7)
axs[5,1].set_xlim([dffig.index.min() - pd.Timedelta(days=2), dffig.index.max() + pd.Timedelta(days=0)])
axs[5,1].set_yticks(np.linspace(ytickmin, ytickmax, ytickstep))
axs[5,1].yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}')) 


# End Figure
labels = ['Model', 'Observation']
fig.legend(bbox_to_anchor=(0.65,0.0005), labels=labels, loc="lower right", ncol=2, fontsize=7, frameon=False)
plt.show()


