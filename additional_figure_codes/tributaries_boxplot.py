# -*- coding: utf-8 -*-
"""
@author: caitlinturner
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import seaborn as sns
from scipy.stats import pearsonr
from dataretrieval import nwis
import matplotlib.ticker as ticker

sns.set_style("ticks")

def process_river_data(site_number, parameter_code, start_date, end_date, conversion_factor, amplification_factor):
    df, _ = nwis.get_dv(sites=site_number, parameterCd=parameter_code, start=start_date, end=end_date)
    df = pd.DataFrame({'discharge': df[f'{parameter_code}_Mean'] * conversion_factor * amplification_factor})
    df = df[~((df.index.month == 2) & (df.index.day == 29))]
    df = df.dropna(subset=['discharge'])
    df['year'] = df.index.year
    df['month'] = df.index.month
    stats_df = df.groupby(['year', 'month'])['discharge'].mean().reset_index()
    stats_df['date'] = pd.to_datetime(stats_df[['year', 'month']].assign(day=1))
    stats_df = stats_df.set_index('date')
    return stats_df

rivers_info = [
    {"name": "Blind River (30% Amite River gauge)", "site_number": "07378500", 
     "conversion_factor": 0.02832, "amplification_factor": 1.41 * 0.30,"y_limits": (0, 0.21)},
    {"name": "Amite River (70% Amite River gauge)", "site_number": "07378500", 
     "conversion_factor": 0.02832, "amplification_factor": 1.41 * 0.70,"y_limits": (0, 0.44)},
    {"name": "Tickfaw River", "site_number": "07376000", 
     "conversion_factor": 0.02832, "amplification_factor": 2.72,"y_limits": (0, 0.21)},
    {"name": "Tchefuncte River", "site_number": "07375000", 
     "conversion_factor": 0.02832, "amplification_factor": 2.02,"y_limits": (0, 0.06)},
    {"name": "Tangipahoa River", "site_number": "07375500", 
     "conversion_factor": 0.02832, "amplification_factor": 1.29,"y_limits": (0, 0.25)}, 
    {"name": "Pearl River", "site_number": "02489500", 
     "conversion_factor": 0.02832, "amplification_factor": 1,"y_limits": (0, 0.25)}, 
    {"name": "Mississippi River",  "site_number": "07374000", 
     "conversion_factor": 0.02832, "amplification_factor": 1,"y_limits": (0, 2.1)},
]

lake_tributaries_discharge = pd.DataFrame()
mississippi_river_discharge = pd.DataFrame()

for river in rivers_info:
    df = process_river_data(river["site_number"], "00060", '2005-01-01', '2022-12-31', 
                            river["conversion_factor"], river["amplification_factor"])
    if river["name"] == "Mississippi River":
        mississippi_river_discharge = df
    else:
        lake_tributaries_discharge = pd.concat([lake_tributaries_discharge, df])

mississippi_river_discharge = mississippi_river_discharge.groupby(['year', 'month']).sum()
lake_tributaries_combined = lake_tributaries_discharge.groupby(['year', 'month']).sum()

# Instead of creating time range, join them directly:
combined = pd.DataFrame({
    't1': mississippi_river_discharge['discharge'],
    't2': lake_tributaries_combined['discharge'],
}).dropna()

# Create proper datetime index
combined['Month'] = [month for _, month in combined.index]
combined.index = pd.to_datetime(['-'.join(map(str, idx)) for idx in combined.index])

# Scale to 10^3 m³/s
combined['t1'] *= 0.001
combined['t2'] *= 0.001

df_bp = combined.copy()

mean_t1 = df_bp.groupby('Month')['t1'].mean()
mean_t2 = df_bp.groupby('Month')['t2'].mean()

std_t1 = df_bp.groupby('Month')['t1'].std()
std_t2 = df_bp.groupby('Month')['t2'].std()


month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}


# Create a DataFrame to hold the statistics
stats_table = pd.DataFrame({
    'Month': [month_map[i] for i in range(1, 13)],
    'Mean Mississippi River Discharge (10^3 m^3/s)': mean_t1.values,
    'Std Dev Mississippi River Discharge (10^3 m^3/s)': std_t1.values,
    'Mean Lake Pontchartrain Tributaries Discharge (10^3 m^3/s)': mean_t2.values,
    'Std Dev Lake Pontchartrain Tributaries Discharge (10^3 m^3/s)': std_t2.values
})


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.48, 2.5), dpi=600, sharex=True) 
fig.subplots_adjust(hspace=0.15)
colors = [cmocean.cm.thermal_r(0.1), cmocean.cm.thermal_r(0.5), cmocean.cm.thermal_r(0.9)] 

month_map = {1: 'J', 2: 'F', 3: 'M', 4: 'A', 5: 'M', 6: 'J', 7: 'J', 8: 'A', 9: 'S', 10: 'O', 11: 'N', 12: 'D'}
df_bp['MonthName'] = df_bp['Month'].map(month_map)
months_ordered = list(month_map.values())
boxplot_data_t1 = [df_bp[df_bp['MonthName'] == month]['t1'].values for month in months_ordered]
boxplot_data_t2 = [df_bp[df_bp['MonthName'] == month]['t2'].values for month in months_ordered]

ax1.text(0.045, 0.88, 'Mississippi River', fontsize=7, transform=ax1.transAxes)
ax1.text(0.01, 0.88, 'a.', fontsize=8, fontweight='bold', transform=ax1.transAxes)
ax2.text(0.045, 0.88, 'Lake Pontchartrain Tributaries', fontsize=7, transform=ax2.transAxes)
ax2.text(0.01, 0.88, 'b.', fontsize=8, fontweight='bold', transform=ax2.transAxes)

fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.ylabel(r'Discharge (10$^3$ m$^{3}$ s$^{-1}$)', labelpad=5, fontsize=8)

flierprops = dict(marker='.', color=colors[1], markersize=5)
bp_t1 = ax1.boxplot(boxplot_data_t1, positions=np.arange(len(months_ordered)), patch_artist=True, flierprops=flierprops)
bp_t2 = ax2.boxplot(boxplot_data_t2, positions=np.arange(len(months_ordered)), patch_artist=True, flierprops=flierprops)
for element in ['boxes']:
    plt.setp(bp_t1[element], color=colors[1])
    plt.setp(bp_t2[element], color=colors[1])   
for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp_t1[element], color='black')
    plt.setp(bp_t2[element], color='black') 

ax1.plot(np.arange(len(months_ordered)), mean_t1.sort_index().values, color=colors[2], marker='o', markersize=3, label='Mean')
ax2.plot(np.arange(len(months_ordered)), mean_t2.sort_index().values, color=colors[2], marker='o', markersize=3, label='Mean Discharge (Lake Pontchartrain Tributaries)')

ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.1f}'))

ax1.set_xticks(np.arange(len(months_ordered)))
ax2.set_xticks(np.arange(len(months_ordered)))
ax2.set_xticklabels([month.upper() for month in months_ordered])  
ax1.tick_params(axis='both', which='major', labelsize=8)
ax2.tick_params(axis='both', which='major', labelsize=8)
ax1.set_ylim(0, 60)
ax2.set_ylim(0, 3)
ax2.set_yticks([0,1,2,3])

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)
plt.show()

# Correlation Coefficient
t1, t2 = lake_tributaries_combined['discharge'], mississippi_river_discharge['discharge']
corr, p_value = pearsonr(t1, t2)

