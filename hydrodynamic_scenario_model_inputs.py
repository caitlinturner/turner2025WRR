# -*- coding: utf-8 -*-
"""
@author: cturn
"""

## Upload Packages ## -------------------------------------------------------
import pandas as pd #good
import matplotlib.pyplot as plt #good
from noaa_coops import Station #good
from datetime import datetime #good
from dateutil.relativedelta import relativedelta #good
from matplotlib.dates import MonthLocator, date2num #good
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
import matplotlib.dates as mdates #good
import cmocean #good
from datetime import datetime, timedelta
import numpy as np #good
from dataretrieval import nwis
from scipy.signal import butter, filtfilt


# Static Inputs
## Set Dates for Model Format Requirements ## -------------------------------
begin = datetime(2020, 12, 1, 0, 0)
dates = []
for i in range(9504):
    dates.append(begin)
    begin += relativedelta(hours=1)
dates = pd.DataFrame({'date': dates,'dateplt': dates})
dates['date'] = dates['date'].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))

begin = datetime(2020, 12, 1, 0, 0)
datesb = []
for i in range(95041):
    datesb.append(begin)
    begin += relativedelta(minutes=6)
datesb = pd.DataFrame({'date': datesb,'dateplt': datesb})
datesb['date'] = datesb['date'].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))


## Download Data ## ---------------------------------------------------------
# Water Level
station_id = "8761305"
df = Station(id=station_id)
df = df.get_data(begin_date="20131201",
                 end_date="20150101",
                 product="water_level",
                 datum="MLLW",
                 units="metric",
                 time_zone="gmt")
df = pd.DataFrame({"predicted": df.v})
df['datetime'] = df.index
valid_data = df['predicted'].values
valid_data = np.nan_to_num(valid_data)  
dtf = (df['datetime'] - df['datetime'].shift()).dt.total_seconds().fillna(0).mean()
Tc = 4 * 3600  # 4 hours in seconds
Wn = 2 * dtf / Tc
B, A = butter(6, Wn, 'low')
waterlevel = filtfilt(B, A, valid_data)
waterlevel = pd.DataFrame({"date": datesb.date,"predicted": waterlevel})
waterlevel['date'] = pd.to_datetime(waterlevel['date'], format='%d/%m/%Y %H:%M')
waterlevel['date_num'] = date2num(waterlevel['date'])

# Wind
df = Station(id="8761927")
df = df.get_data(begin_date="20131201",
                                   end_date="20150101",
                                    product="wind",
                                    units="metric",
                                    time_zone="gmt")
wind = pd.DataFrame({"speed": df.s,"direction": df.d})
wind = wind.resample('h').mean()
wind.reset_index(inplace=True)
wind = wind[:-1]
wind = pd.DataFrame({"date": dates.date,"speed": wind.speed, 'direction': wind.direction})
wind['date'] = pd.to_datetime(wind['date'], format='%d/%m/%Y %H:%M')
wind['date_num'] = date2num(wind['date'])


# Pressure
df = Station(id="8761927")
df = df.get_data(begin_date="20131201",
                                   end_date="20150101",
                                    product="air_pressure",
                                    units="metric",
                                    time_zone="gmt")

pressure = pd.DataFrame({"air": df.v})
pressure = pressure.resample('h').mean()
pressure.reset_index(inplace=True)
pressure = pressure[:-1]
pressure = pd.DataFrame({"date": dates.date,"pressure": pressure.air})
pressure['date'] = pd.to_datetime(pressure['date'], format='%d/%m/%Y %H:%M')
pressure['date_num'] = date2num(pressure['date'])

data = {
    'timestamp': wind.date,
    'pressure': pressure.pressure,
    'wind_speed': wind.speed,  
    'wind_direction': wind.direction  
}
df = pd.DataFrame(data)

downsampled_data = df.resample('24h', on='timestamp').agg({
    'pressure': 'mean',
    'wind_speed': 'mean',
    'wind_direction': lambda x: (np.rad2deg(np.arctan2(np.mean(np.sin(np.deg2rad(x))), np.mean(np.cos(np.deg2rad(x))))) + 360) % 360
})



# Dynamic Inputs
## Total Discharge Data ## -----------------------------------------------
def generate_dates(start_year, start_month, num_days):
    begin_date = datetime(start_year, start_month, 1)
    end_date = begin_date + timedelta(days=num_days)
    dates = pd.date_range(start=begin_date, end=end_date, freq='ME')
    return pd.DataFrame({'date': dates})

def generate_dates2(start_year, start_month, num_days):
    begin_date = datetime(start_year, start_month, 1)
    end_date = begin_date + timedelta(days=num_days)
    dates = pd.date_range(start=begin_date, end=end_date, freq='ME')
    return pd.DataFrame({'date': dates})


# Retrieve and process river data
def process_river_data(site_number, parameter_code, start_date, end_date, conversion_factor, amplification_factor):
    df, _ = nwis.get_dv(sites=site_number, parameterCd=parameter_code, start=start_date, end=end_date)
    df = pd.DataFrame({'discharge': df[f'{parameter_code}_Mean'] * conversion_factor * amplification_factor*0.001})
    df = df[~((df.index.month == 2) & (df.index.day == 29))]
    df['dates'] = df.index
    stats_df = df.groupby(df.index.month)['discharge'].agg(mean='mean', p95th=lambda x: np.nanpercentile(x, 95), p5th=lambda x: np.nanpercentile(x, 5)).reset_index()
    stats_df['dates'] = dates_df2
    stats_df = stats_df.set_index('dates')
    first_row = stats_df.iloc[0]
    last_row = stats_df.iloc[-1]
    stats_df = pd.concat([last_row.to_frame().T, stats_df, first_row.to_frame().T], ignore_index=True)
    stats_df['datetime'] = np.arange(0.5, 14.5, 1)
    return stats_df


# Generate dates
dates_df = generate_dates(2021, 1, 12)
dates_df2 = generate_dates2(2021, 1, 12)


# Process each river
rivers_info = [
    {"name": "Amite River (30% Amite River gauge)", "site_number": "07378500", 
     "conversion_factor": 0.02832, "amplification_factor": 1.41 * 0.30,"y_limits": (0, 0.21)},
    {"name": "Blind River (70% Amite River gauge)", "site_number": "07378500", 
     "conversion_factor": 0.02832, "amplification_factor": 1.41 * 0.70,"y_limits": (0, 0.42)},
    {"name": "Tickfaw River", "site_number": "07376000", 
     "conversion_factor": 0.02832, "amplification_factor": 2.72,"y_limits": (0, 0.21)},
    {"name": "Tchefuncte River", "site_number": "07375000", 
     "conversion_factor": 0.02832, "amplification_factor": 2.02,"y_limits": (0, 0.06)},
    {"name": "Tangipahoa River", "site_number": "07375500", 
     "conversion_factor": 0.02832, "amplification_factor": 1.29,"y_limits": (0, 0.25)},
    {"name": "Pearl River",  "site_number": "02489500", 
     "conversion_factor": 0.02832, "amplification_factor": 1.35,"y_limits": (0, 2.1)},]

river_data_for_excel = {}

for river in rivers_info:
    df = process_river_data(river["site_number"], "00060", '1930-12-31', '2023-12-31', 
                            river["conversion_factor"], river["amplification_factor"])
    river_data_for_excel[river["name"]] = df



## Total Discharge Figure ## -----------------------------------------------
def aggregate_total_discharge(river_data_dict):
    total_discharge_df = None
    for river_name, df in river_data_dict.items():
        if total_discharge_df is None:
            total_discharge_df = df.copy()
        else:
            total_discharge_df['p95th'] += df['p95th']
            total_discharge_df['mean'] += df['mean']
            total_discharge_df['p5th'] += df['p5th']
    return total_discharge_df

total_discharge_df = aggregate_total_discharge(river_data_for_excel)




## Synthetic Spillway ## -----------------------------------------------------
# Data from USACE
bcs = pd.read_excel(r'hydrosharedata/data/Bonnet_Carre_ALL_Weeks_USACE.xlsx')
bcs['date'] = pd.to_datetime(bcs['date'])  
bcs['date'] = bcs['date'].dt.strftime("%m/%d")
bcs.set_index('date', inplace=True)
year = ['2020','2019a','2019b','2018','2016','2011','2008','1997','1983','1979','1975','1973','1950','1945','1937']
bcs = bcs.map(lambda x: x*0.001 if pd.notna(x) else x)
yearday = list(range(1, 214))
bcsstat = bcs.copy()
bcsstat.replace(0, pd.NA, inplace=True)
bcsstat['yearday'] = yearday
bcsstat.set_index('yearday', inplace=True)

results = []
for column in bcsstat.columns:
    daysopen = bcsstat[column].count()
    maxdischarge = bcsstat[column].max()
    totaldischarge = bcsstat[column].sum()
    startdate = bcsstat[column].first_valid_index()
    enddate = bcsstat[column].last_valid_index()

    results.append({
        'Year': column,
        'Days Open': daysopen,
        'Maximum Discharge': maxdischarge,
        'Total Discharge': totaldischarge,
        'Yearday Open': startdate,
        'Date Closed': enddate})

results = pd.DataFrame(results)
averages = pd.DataFrame({'daysopen': [int(results['Days Open'].mean())],
                          'maxdischarge': [int(results['Maximum Discharge'].mean())],
                          'totaldischarge': [int(results['Total Discharge'].mean())],
                          'yearday': [int(results['Yearday Open'].mean())]})
auc = averages.totaldischarge.item()
def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
		denom = (x1 - x2) * (x1 - x3) * (x2 - x3);
		A     = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom;
		B     = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom;
		C     = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom;
		return A,B,C
x1, y1 = [averages.yearday.item() - 1, 0]
x2, y2 = [averages.yearday.item() + (averages.daysopen.item()/2), averages.maxdischarge.item()*0.98]
x3, y3 = [averages.yearday.item() + averages.daysopen.item() + 1, 0]
a, b, c = calc_parabola_vertex(x1, y1, x2, y2, x3, y3)
xloc = np.arange(x1,x3+1,1)
yloc = []
for x in range(len(xloc)):
 	xval = xloc[x]
 	y = (a*(xval**2)) + (b*xval) + c
 	yloc.append(y)



## Figure ## ---------------------------------------------------------
def uppercase_first_letter_month(x, pos):
    return mdates.num2date(x).strftime('%b')[0].upper()

xx = 0.045
# Colors used
colors_used = [cmocean.cm.thermal_r(0.1), cmocean.cm.thermal_r(0.5), cmocean.cm.thermal_r(0.9)]
colors = [cmocean.cm.thermal_r(x) for x in [0.1, 0.25, 0.5, 0.75, 0.9]]
fig, axs = plt.subplots(4, 1, figsize=(4.5, 4), dpi=600)  # 4 rows, 1 column
plt.subplots_adjust(hspace=0.3)


# Plot 1 - Wind speed 
axs[0].plot(downsampled_data.index, downsampled_data['wind_speed'], color=colors[3], linewidth=0.5)
axs[0].set_xlim([date2num(pd.to_datetime('2020-12-30')), date2num(pd.to_datetime('2021-12-31'))])
axs[0].set_ylim([0, 13])
axs[0].set_yticks([0, 5, 10])
axs[0].xaxis.set_major_locator(MonthLocator())
axs[0].xaxis.set_major_formatter(FuncFormatter(uppercase_first_letter_month))
axs[0].tick_params(axis='both', which='major', labelsize=8)
axs[0].set_ylabel('Wind speed\n(m s$^{-1}$)', fontsize=8, labelpad=3)
axs[0].text(xx, 0.95, 'a.', transform=axs[0].transAxes, fontsize=8, fontweight='bold', va='top', ha='right')
axs[0].text(xx + 0.005, 0.95, 'Wind conditions', transform=axs[0].transAxes, fontsize=8, va='top', ha='left')
axs[0].set_xticklabels([])  

# Plot 2 - Water level 
axs[1].plot(waterlevel['date'], waterlevel['predicted'], color=colors[4], linewidth=0.5)
axs[1].set_xlim([date2num(pd.to_datetime('2020-12-30')), date2num(pd.to_datetime('2021-12-31'))])
axs[1].xaxis.set_major_locator(MonthLocator())
axs[1].set_yticks([0, 0.5, 1.0])
axs[1].set_ylim([-0.4, 1.4])
axs[1].xaxis.set_major_formatter(FuncFormatter(uppercase_first_letter_month))
axs[1].tick_params(axis='y', which='major', labelsize=8)
axs[1].set_ylabel('Water level\nMLLW (m)', fontsize=8, labelpad=2.5)
axs[1].text(xx, 0.95, 'b.', transform=axs[1].transAxes, fontsize=8, fontweight='bold', va='top', ha='right')
axs[1].text(xx + 0.005, 0.95, 'Ocean boundary', transform=axs[1].transAxes, fontsize=8, va='top', ha='left')
axs[1].set_xticklabels([]) 

# Plot 3 - Spillway
linestyles = ['--', '-', ':']
axs[2].plot(xloc, yloc, linestyle='-', color='black', alpha=0.6, linewidth=1.5)
axs[2].scatter(x1, y1, marker="s", s=20, color=colors[2], edgecolors='black', 
                  label='$(x_0, y_0)$ = (MAR 26, 0)', zorder=3)
axs[2].scatter(x2, y2, marker="o", s=20, color=colors[3], edgecolors='black',
                  label='$(x_1, y_1)$ = (APR 14, 6054)', zorder=3)
axs[2].scatter(x3, y3, marker="P", s=20, color=colors[0], edgecolors='black',
                  label='$(x_2, y_2)$ = (MAY 3, 0)', zorder=3)
axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axs[2].axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
axs[2].set_xlim(-1, 366)
axs[2].set_xticks([1, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
axs[2].set_xticklabels([])
axs[2].tick_params(axis='y', labelsize=8)
axs[2].legend(loc='upper left', frameon=False, prop={'size': 8}, bbox_to_anchor=(0.50, 1.10), labelspacing=0.1, handletextpad=0.2)
axs[2].set_ylim(-0.5, 9.2)
axs[2].set_yticks([0, 3, 6,9])
axs[2].set_ylabel('Discharge\n(10$^3$ m$^{3}$ s$^{-1}$)', fontsize=8, labelpad=3)
axs[2].text(xx, 0.95, 'c.', transform=axs[2].transAxes, fontsize=8, fontweight='bold', va='top', ha='right')
axs[2].text(xx + 0.005, 0.95, 'Diversion opening', transform=axs[2].transAxes, fontsize=8, va='top', ha='left')
axs[2].set_xticklabels([])  


# Plot 4 - Tributary 
axs[3].plot(total_discharge_df['datetime'], total_discharge_df['p95th'], label='High', color=colors[2], linestyle=linestyles[0], linewidth=1.5)
axs[3].plot(total_discharge_df['datetime'], total_discharge_df['mean'], label='Median', color=colors[3], linestyle=linestyles[1], linewidth=1.5)
axs[3].plot(total_discharge_df['datetime'], total_discharge_df['p5th'], label='Low', color=colors[0], linestyle=linestyles[2], linewidth=1.5)
axs[3].set_xlim(0.95, 13.05)
axs[3].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
axs[3].set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], fontsize = 8)
axs[3].tick_params(axis='y', labelsize=8)
axs[3].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axs[3].set_ylim(0, 4.5)
axs[3].set_yticks([0, 2, 4])
axs[3].legend(loc='upper left', ncol=1, prop={'size': 8}, bbox_to_anchor=(0.50, 1.10), frameon=False, labelspacing=0.2, handletextpad=0.5)
axs[3].set_ylabel('Discharge\n(10$^3$ m$^{3}$ s$^{-1}$)', fontsize=8, labelpad=3)
axs[3].text(xx, 0.95, 'd.', transform=axs[3].transAxes, fontsize=8, fontweight='bold', va='top', ha='right')
axs[3].text(xx + 0.005, 0.95, 'Tributary conditions', transform=axs[3].transAxes, fontsize=8, va='top', ha='left')

plt.show()