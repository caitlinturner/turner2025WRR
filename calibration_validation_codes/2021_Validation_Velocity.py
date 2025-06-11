import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from dataretrieval import nwis
from noaa_coops import Station
import seaborn as sns
from goodness_of_fit import d, r_pearson, rmse
from datetime import datetime
from matplotlib.gridspec import GridSpec
import cmocean
import warnings
warnings.filterwarnings("ignore")

## Import Model Data 
dir_testinput = r'Validation_Velocity\2021_model_review_updates.dsproj_data\FlowFM\output'
file_nc_his = os.path.join(dir_testinput, 'FlowFM_his.nc')

## Dates
sd = '2021-10-05'
ed = '2021-10-30'
sdm = '2021-10-05 00:00:00'
edm = '2021-10-30 23:00:00'
begin = datetime(2021, 10, 5, 0, 0)
dates = pd.date_range(start=begin, end=datetime(2021, 10, 30, 23, 0), freq='H')
dates_df = pd.DataFrame({'date': dates})
dates_df['date'] = dates_df['date'].dt.strftime('%d/%m/%Y %H:%M')

## Model Data
move_mean = 2
data_xr_his = xr.open_dataset(file_nc_his)
station_names = data_xr_his.station_name.astype(str).to_pandas()
df = data_xr_his.waterlevel.to_pandas()
df.columns = station_names
df.index = df.index - pd.DateOffset(hours=0)

## Stats Table Set Up
stats_table = pd.DataFrame(columns=[
    'Location', 'Index of Agreement', 'Pearson Corr Coeff', 'Root Mean Squared Error'])

## Figure Setup
ytickmin, ytickmax, ytickstep = -0.5, 2.2, 4
ymin, ymax = -0.5, 2.2
xstattxt, ystattxt = 0.34, 0.5
xx, yy = 0.04, 0.58
lw = 1.25
sns.set_palette("colorblind")
colors = [cmocean.cm.thermal_r(0.9), cmocean.cm.thermal_r(0.5)]
linestyles_used = ['-', '--']

fig = plt.figure(figsize=(6.5, 3), dpi=600)
gs = GridSpec(nrows=4, ncols=4,
              width_ratios=[1, 1, 0.2, 1],
              wspace=0.05,
              hspace=0.3)
axs = np.empty((4, 3), dtype=object)
for i in range(4):
    axs[i, 0] = fig.add_subplot(gs[i, 0])

for i in range(4):
    axs[i, 1] = fig.add_subplot(gs[i, 1])

for i in range(4):
    axs[i, 2] = fig.add_subplot(gs[i, 3])

fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.ylabel('Water level (m above MLLW)', labelpad=8, fontsize=6)

## Water Level Open-Source
# New Canal (NOAA)
df_noaa_newcanal = Station(id="8761927").get_data(
    begin_date="20211005", end_date="20211030",
    product="hourly_height", datum="MLLW",
    units="metric", time_zone="gmt")
obs = df_noaa_newcanal.v[sd:ed].reset_index(drop=True)
model = df.NewCanal_USGS[sdm:edm].reset_index(drop=True)
datum_diff = (model - obs).mean()
model = model - datum_diff

common_index = model.index.intersection(obs.index)
model = model.loc[common_index]
obs = obs.loc[common_index]

dffig = pd.DataFrame({'date': dates_df.date, 'Model': model, 'Obs': obs})
dffig['date'] = pd.to_datetime(dffig['date'], format="%d/%m/%Y %H:%M")
dffig.set_index('date', inplace=True)
dffig = dffig.resample('D').mean().rolling(move_mean, min_periods=1).mean()

obs_clean = dffig['Obs'].dropna()
model_clean = dffig['Model'][obs_clean.index].dropna()
common_index = obs_clean.index.intersection(model_clean.index)
obs_clean = obs_clean.loc[common_index]
model_clean = model_clean.loc[common_index]
stats_table.loc[len(stats_table)] = [
    'New Canal (NOAA: 8761927)',
    round(d(obs_clean, model_clean), 2),
    round(r_pearson(obs_clean, model_clean), 2),
    round(rmse(obs_clean, model_clean), 2),]

for i, series in enumerate(['Model', 'Obs']):
    axs[0, 0].plot(dffig.index, dffig[series],
                   linewidth=lw, color=colors[i], label=series,
                   linestyle=linestyles_used[i])
axs[0, 0].set_ylim([ymin, ymax])
axs[0, 0].tick_params(axis='y', labelsize=6)
axs[0, 0].set_title('a.', fontsize=6,
                    loc='center', x=xx, y=yy+0.05)
axs[0, 0].text(xstattxt+0.1, ystattxt+0.08,
               f'$d$ = {stats_table.iloc[0, 1]:.2f}, '
               f'r$^2$ = {stats_table.iloc[0, 2]:.2f},\n'
               f'RMSE = {stats_table.iloc[0, 3]:.2f}',
               transform=axs[0, 0].transAxes, fontsize=6)
axs[0, 0].set_yticks(np.linspace(ytickmin, ytickmax, ytickstep))
axs[0, 0].yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))
axs[0, 0].text(xx + 0.025, yy +0.04, 'New Canal\n(NOAA)',
               transform=axs[0, 0].transAxes, fontsize=6)

# Shell Beach (NOAA)
df_noaa_shellbeach = Station(id="8761305").get_data(
    begin_date="20211005", end_date="20211030",
    product="hourly_height", datum="MLLW",
    units="metric", time_zone="gmt")
obs = df_noaa_shellbeach.v[sd:ed].reset_index(drop=True)
model = df.ShellBeach_NOAA[sdm:edm].reset_index(drop=True)
datum_diff = (model - obs).mean()
model = model - datum_diff

common_index = model.index.intersection(obs.index)
model = model.loc[common_index]
obs = obs.loc[common_index]

dffig = pd.DataFrame({'date': dates_df.date, 'Model': model, 'Obs': obs})
dffig['date'] = pd.to_datetime(dffig['date'], format="%d/%m/%Y %H:%M")
dffig.set_index('date', inplace=True)
dffig = dffig.resample('D').mean().rolling(move_mean, min_periods=1).mean()

obs_clean = dffig['Obs'].dropna()
model_clean = dffig['Model'][obs_clean.index].dropna()
common_index = obs_clean.index.intersection(model_clean.index)
obs_clean = obs_clean.loc[common_index]
model_clean = model_clean.loc[common_index]
stats_table.loc[len(stats_table)] = [
    'Shell Beach \n (NOAA: 8761305)',
    round(d(obs_clean, model_clean), 2),
    round(r_pearson(obs_clean, model_clean), 2),
    round(rmse(obs_clean, model_clean), 2),]

for i, series in enumerate(['Model', 'Obs']):
    axs[1, 0].plot(dffig.index, dffig[series],
                   linewidth=lw, color=colors[i], label=series,
                   linestyle=linestyles_used[i])
axs[1, 0].set_ylim([ymin, ymax])
axs[1, 0].tick_params(axis='y', labelsize=6)
axs[1, 0].set_title('b.', fontsize=6,
                    loc='center', x=xx, y=yy+0.05)
axs[1, 0].text(xstattxt+0.1, ystattxt+0.08,
               f'$d$ = {stats_table.iloc[1, 1]:.2f}, '
               f'r$^2$ = {stats_table.iloc[1, 2]:.2f},\n'
               f'RMSE = {stats_table.iloc[1, 3]:.2f}',
               transform=axs[1, 0].transAxes, fontsize=6)
axs[1, 0].set_yticks(np.linspace(ytickmin, ytickmax, ytickstep))
axs[1, 0].yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))
axs[1, 0].text(xx + 0.025, yy +0.04, 'Shell Beach\n(NOAA)',
               transform=axs[1, 0].transAxes, fontsize=6)

# The Rigolets (USGS)
parameterCode = '00065'
site = '301001089442600'
df_usgs_rigolets = nwis.get_record(
    sites=site, service='iv', parameterCd=parameterCode,
    start='2021-10-05', end='2021-10-30'
)
df_usgs_rigolets = df_usgs_rigolets.rename(
    columns={'00065': 'waterlevel', '00065_cd': 'cd', 'site_no': 'site'}
)
obs = df_usgs_rigolets['waterlevel'].resample('H').mean().multiply(0.3048)
obs = obs[sd:ed].tz_localize(None)

model = df.Rigolets_USGS[sdm:ed]
common_index = model.index.intersection(obs.index)
model = model.loc[common_index]
obs = obs.loc[common_index]

datum_diff = (model - obs).mean()
model = model - datum_diff

dffig = pd.DataFrame({'Model': model, 'Obs': obs})
dffig = dffig.resample('D').mean().rolling(move_mean, min_periods=1).mean()

obs_clean = dffig['Obs'].dropna()
model_clean = dffig['Model'][obs_clean.index].dropna()
common_index = obs_clean.index.intersection(model_clean.index)
obs_clean = obs_clean.loc[common_index]
model_clean = model_clean.loc[common_index]
stats_table.loc[len(stats_table)] = [
    f'The Rigolets \n (USGS: {site})',
    round(d(obs_clean, model_clean), 2),
    round(r_pearson(obs_clean, model_clean), 2),
    round(rmse(obs_clean, model_clean), 2),]

for i, series in enumerate(['Model', 'Obs']):
    axs[2, 0].plot(dffig.index, dffig[series],
                   linewidth=lw, color=colors[i], label=series,
                   linestyle=linestyles_used[i])
axs[2, 0].set_ylim([ymin, ymax])
axs[2, 0].tick_params(axis='y', labelsize=6)
axs[2, 0].set_title('c.', fontsize=6,
                    loc='center', x=xx, y=yy+0.05)
axs[2, 0].text(xstattxt+0.1, ystattxt+0.08,
               f'$d$ = {stats_table.iloc[2, 1]:.2f}, '
               f'r$^2$ = {stats_table.iloc[2, 2]:.2f},\n'
               f'RMSE = {stats_table.iloc[2, 3]:.2f}',
               transform=axs[2, 0].transAxes, fontsize=6)
axs[2, 0].set_yticks(np.linspace(ytickmin, ytickmax, ytickstep))
axs[2, 0].yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))
axs[2, 0].text(xx + 0.025, yy +0.04, 'Rigolets\n(USGS)',
               transform=axs[2, 0].transAxes, fontsize=6)

# Crossover (USGS)
site = '301200090072400'
df_usgs_cross = nwis.get_record(
    sites=site, service='iv', parameterCd=parameterCode,
    start='2021-10-05', end='2021-10-30'
)
df_usgs_cross = df_usgs_cross.rename(
    columns={'00065': 'waterlevel', '00065_cd': 'cd', 'site_no': 'site'}
)
obs = df_usgs_cross['waterlevel'].resample('H').mean().multiply(0.3048)
obs = obs[sd:ed].tz_localize(None)

model = df.Causeway_USGS[sdm:ed]
common_index = model.index.intersection(obs.index)
model = model.loc[common_index]
obs = obs.loc[common_index]

datum_diff = (model - obs).mean()
model = model - datum_diff

dffig = pd.DataFrame({'Model': model, 'Obs': obs})
dffig = dffig.resample('D').mean().rolling(move_mean, min_periods=1).mean()

obs_clean = dffig['Obs'].dropna()
model_clean = dffig['Model'][obs_clean.index].dropna()
common_index = obs_clean.index.intersection(model_clean.index)
obs_clean = obs_clean.loc[common_index]
model_clean = model_clean.loc[common_index]
stats_table.loc[len(stats_table)] = [
    f'Crossover \n (USGS: {site})',
    round(d(obs_clean, model_clean), 2),
    round(r_pearson(obs_clean, model_clean), 2),
    round(rmse(obs_clean, model_clean), 2),
]

for i, series in enumerate(['Model', 'Obs']):
    axs[3, 0].plot(dffig.index, dffig[series],
                   linewidth=lw, color=colors[i], label=series,
                   linestyle=linestyles_used[i])
axs[3, 0].set_ylim([ymin, ymax])
axs[3, 0].tick_params(axis='y', labelsize=6)
axs[3, 0].set_title('d.', fontsize=6,
                    loc='center', x=xx, y=yy+0.05)
axs[3, 0].text(xstattxt+0.1, ystattxt+0.08,
               f'$d$ = {stats_table.iloc[3, 1]:.2f}, '
               f'r$^2$ = {stats_table.iloc[3, 2]:.2f},\n'
               f'RMSE = {stats_table.iloc[3, 3]:.2f}',
               transform=axs[3, 0].transAxes, fontsize=6)
axs[3, 0].set_yticks(np.linspace(ytickmin, ytickmax, ytickstep))
axs[3, 0].yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))
axs[3, 0].text(xx + 0.025, yy +0.04, 'Crossover\n(USGS)',
               transform=axs[3, 0].transAxes, fontsize=6)

# LSU MOORINGS Water Level
xx, yy = 0.04, 0.39
stations = {
    'Rigolets\n(LSU)':     {'sheet': 'Rigolets',    'model_station': 'Rigolets_CRRT'},
    'North Shore\n(LSU)':   {'sheet': 'NorthShore',  'model_station': 'NorthShore_CRRT'},
    ' BCS\n(LSU)':  {'sheet': 'BonneCarre',  'model_station': 'BCS_CRRT'},
    'Maurepas\n(LSU)':     {'sheet': 'Maurepas',    'model_station': 'Maurepas_CRRT'},
}
sdm = '2021-10-05 00:00:00'
edm = '2021-10-30 23:00:00'
resample_rule = '24H'
rolling_window = 1

df_wl = data_xr_his.waterlevel.to_pandas()
df_wl.columns = station_names
df_wl.index = df_wl.index - pd.DateOffset(hours=0)

for i, (label, info) in enumerate(stations.items()):
    # Load observed
    obs_df = pd.read_excel("data/Mooring_data_2021.xlsx", sheet_name=info['sheet'])
    obs_df['Date'] = pd.to_datetime(obs_df['Date'])
    obs = obs_df[['Date', 'Water_Level_MLLW']].rename(columns={'Water_Level_MLLW': 'Level'})
    obs = obs.set_index('Date').resample(resample_rule).mean().rolling(rolling_window, center=True).mean().reset_index()
    obs = obs[(obs['Date'] >= sdm) & (obs['Date'] <= edm)]

    # Load model
    model = df_wl[info['model_station']][sdm:edm].reset_index()
    model.columns = ['Date', 'Level']
    model = model.set_index('Date').resample(resample_rule).mean().rolling(rolling_window, center=True).mean().reset_index()

    mean_model = model['Level'].mean()
    mean_obs = obs['Level'].mean()
    if label in ['Rigolets', 'BonnetCarre']:
        mean_diff = mean_model - mean_obs
    else:
        mean_diff = mean_model - mean_obs

    print(f"{label} means — Model: {mean_model:.3f} m, Obs: {mean_obs:.3f} m, Diff: {mean_diff:.3f} m")

    model['Level'] = model['Level'] - mean_diff

    combined = pd.merge(obs, model, on='Date', how='inner', suffixes=('_obs', '_model')).dropna()
    obs_clean = combined['Level_obs'].to_numpy()
    model_clean = combined['Level_model'].to_numpy()
    d_val = round(d(model_clean, obs_clean), 2)
    r_val = round(r_pearson(model_clean, obs_clean), 2)
    rmse_val = round(rmse(model_clean, obs_clean), 2)

    axs[i, 1].plot(model['Date'], model['Level'], linewidth=lw, color=colors[0], linestyle=linestyles_used[0], label='Model')
    axs[i, 1].plot(obs['Date'], obs['Level'], linewidth=lw, color=colors[1], linestyle=linestyles_used[1], label='Obs')
    axs[i, 1].set_ylim([ytickmin, ytickmax])
    axs[i, 1].set_yticks([])
    axs[i, 1].yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))
    axs[i, 1].set_title(f'{chr(101+i)}.', fontsize=6, loc='center', x=xx, y=yy+0.25)
    axs[i, 1].text(xx + 0.025, yy + 0.225, label, transform=axs[i, 1].transAxes, fontsize=6)

    axs[i, 1].text(xstattxt+0.1, ystattxt+0.08,
                   f'$d$ = {d_val:.2f}, r$^2$ = {r_val:.2f},\nRMSE = {rmse_val:.2f}',
                   transform=axs[i, 1].transAxes, fontsize=6)

fig.legend(
    bbox_to_anchor=(0.59, -0.02),
    labels=['Model', 'Observation'],
    loc="lower right",
    ncol=2,
    fontsize=6,
    frameon=False)


## LSU MOORINGS Velocity Magnitude 
sdm = '2021-10-05 00:00:00'
edm = '2021-10-30 23:00:00'
rolling_window = 12
lw = 1
xx, yy = 0.04, 0.39
xstattxt, ystattxt = 0.22, 0.8
ytickmin, ytickmax, ytickstep = 0, 1.5, 4
ymin, ymax = 0, 1.5

df_u = data_xr_his.x_velocity.to_pandas()
df_u.columns = station_names
df_u.index = df_u.index - pd.DateOffset(hours=0)
df_v = data_xr_his.y_velocity.to_pandas()
df_v.columns = station_names
df_v.index = df_v.index - pd.DateOffset(hours=0)

for i, (label, info) in enumerate(stations.items()):
    obs_df = pd.read_excel("data/Mooring_data_2021.xlsx", sheet_name=info['sheet'])
    obs_df['Date'] = pd.to_datetime(obs_df['Date'])
    obs = obs_df[['Date', 'Velocity_Magnitude']].rename(columns={'Velocity_Magnitude': 'Magnitude'})
    obs = obs.set_index('Date').resample(resample_rule).mean().rolling(rolling_window, center=True).mean().reset_index()
    obs = obs[(obs['Date'] >= sdm) & (obs['Date'] <= edm)]
    
    u = df_u[info['model_station']][sdm:edm].reset_index(drop=True)
    v = df_v[info['model_station']][sdm:edm].reset_index(drop=True)
    velocity_magnitude = np.sqrt(u**2 + v**2)
    model_dates = obs['Date'].iloc[:len(velocity_magnitude)]
    model = pd.DataFrame({
        'Date': model_dates,
        'Magnitude': velocity_magnitude[:len(model_dates)]})
    model = model.set_index('Date').mean().rolling(rolling_window, center=True).mean().reset_index()

    combined = pd.merge(obs, model, on='Date', how='inner', suffixes=('_obs', '_model')).dropna()
    obs_clean = combined['Magnitude_obs'].to_numpy()
    model_clean = combined['Magnitude_model'].to_numpy()
    print(r'Station [i]')
    print(model_clean.mean())
    print(model_clean.min())
    print(model_clean.max())
    print(model_clean.max()-model_clean.min())
    d_val = round(d(model_clean, obs_clean), 2)
    r_val = round(r_pearson(model_clean, obs_clean), 2)
    rmse_val = round(rmse(model_clean, obs_clean), 2)

    axs[i, 2].plot(model['Date'], model['Magnitude'], linewidth=lw, color=colors[0], linestyle=linestyles_used[0], label='Model')
    axs[i, 2].plot(obs['Date'], obs['Magnitude'], linewidth=lw, color=colors[1], linestyle=linestyles_used[1], label='Obs')
    axs[i, 2].set_ylim([ymin, ymax])
    axs[i, 2].set_yticks(np.linspace(ytickmin, ytickmax, ytickstep))
    axs[i, 2].yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))
    axs[i, 2].set_title(f'{chr(105+i)}.', fontsize=6, loc='center', x=xx, y=yy+0.25)
    axs[i, 2].text(xx + 0.025, yy + 0.225, label, transform=axs[i, 2].transAxes, fontsize=6)
    axs[2, 2].set_ylabel('                  Water velocity magnitude (m/s)', labelpad=3, fontsize=6)

    axs[i, 2].text(xstattxt+0.21, ystattxt-0.22,
                   f'$d$ = {d_val:.2f}, r$^2$ = {r_val:.2f},\nRMSE = {rmse_val:.2f}',
                   transform=axs[i, 2].transAxes, fontsize=6)
    axs[i,2].tick_params(axis='y', labelsize=6)




## Wrapper
start_date = pd.to_datetime('2021-10-04')
end_date = pd.to_datetime('2021-10-31')
locator = AutoDateLocator()
formatter = ConciseDateFormatter(locator)
formatter.formats = ['%d', '%d', '%d', '%d', '%d', '%d']
formatter.zero_formats = [''] * 6
formatter.offset_formats = [''] * 6

for row in range(4):
    for col in range(3):
        ax = axs[row, col]

        if row < 3:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlim([start_date, end_date])
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            ax.tick_params(axis='x', labelsize=6)
            if col == 0:
                ax.text(-0.15, -0.39, ' OCT\n2021', transform=ax.transAxes,
                        fontsize=6, ha='left', va='center')

plt.show()



