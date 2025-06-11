# -*- coding: utf-8 -*-
"""

@author: cturn
"""
## Upload Packages ## -------------------------------------------------------
import pandas as pd
from matplotlib import pyplot as plt
from noaa_coops import Station
from sklearn.metrics import r2_score
import numpy as np
import cmocean

## Download Data ## ---------------------------------------------------------
df = Station(id="8761927")
df = df.get_data(begin_date="20070101",
                                    end_date="20240101",
                                    product="wind",
                                    units="metric",
                                    time_zone="gmt")


## Daily Stats For All Years ## ----------------------------------------------
wind = pd.DataFrame({"speed": df.s,"direction": df.d})
wind = wind.resample('d').mean()
wind['speedavg'] = wind['speed'].rolling(window=40, min_periods=1).mean()
wind['yearday'] = wind.index.dayofyear
mean_yearday = wind.groupby('yearday')['speedavg'].mean()
mean_yearday = mean_yearday.drop(mean_yearday.index[61])


# RMSE calculation
diffyear = None
mindiff = float('inf')
diffresults = []
for year in range(2007, 2023):
    year_data = wind[wind.index.year == year]
    year_mean = year_data.groupby('yearday')['speedavg'].mean()
    diff = np.sqrt(((mean_yearday - year_mean) ** 2).mean())
    diffresults.append((year, diff))
    if diff < mindiff:
        mindiff = diff
        diffyear = year


# R-Squared calculation
r2year = None
maxr2 = -float('inf')
r2results = []
for year in range(2007, 2023):
    year_data = wind[wind.index.year == year]
    year_mean = year_data.groupby('yearday')['speedavg'].mean()
    non_nan_indices = ~mean_yearday.isna() & ~year_mean.isna()
    r2 = r2_score(mean_yearday[non_nan_indices], year_mean[non_nan_indices])
    r2results.append((year, r2))
    if r2 > maxr2:
        maxr2 = r2
        r2year = year

# Figure
num_colors = 18
color_values = np.linspace(0.2, 1, num_colors)
colors = [cmocean.cm.thermal_r(val) for val in color_values]

fig, axs = plt.subplots(3, 1, figsize=(4.48, 3.5), dpi=600, gridspec_kw={'height_ratios': [1, 1, 1]})

ax_left_top = axs[0]
ax_left_bottom = axs[2]
ax_right = axs[1]

ax_left_top.text(0.035, 0.98, 'a.', transform=ax_left_top.transAxes, fontsize=8, fontweight='bold', va='top', ha='right')
ax_right.text(0.035, 0.95, 'b.', transform=ax_right.transAxes, fontsize=8, fontweight='bold', va='top', ha='right')
ax_left_bottom.text(0.035, 0.98, 'c.', transform=ax_left_bottom.transAxes, fontsize=8, fontweight='bold', va='top', ha='right')

# Wind speed historical analysis
for i, year in enumerate(range(2007, 2023)):
    year_data = wind[wind.index.year == year]
    ax_left_top.plot(year_data['yearday'], year_data['speedavg'], color=colors[i], alpha=0.3)
ax_left_top.plot(mean_yearday.index, mean_yearday, label='Mean', color=colors[17], linewidth=1.5, linestyle='--')
ax_left_top.set_ylabel('Wind speed \n(m s$^{-1}$)', fontsize=8)
ax_left_top.set_xticks([3, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
ax_left_top.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], fontsize=8)
ax_left_top.set_yticks([2.0, 4.0, 6.0])
ax_left_top.set_yticklabels([2.0, 4.0, 6.0], fontsize=8)
ax_left_top.set_xlim([0, 365])
ax_left_top.set_ylim([1.5, 6.1])
ax_left_top.legend(loc="upper right", fontsize=8, frameon=False, bbox_to_anchor=(1.01, 1.065))
ax_left_top.tick_params(axis='x', labelsize=8)
ax_left_top.tick_params(axis='y', labelsize=8)

# RMSE and R-squared 
ax1 = ax_right
ax2 = ax1.twinx()
line1 = ax1.plot([result[0] for result in r2results],
                 [result[1] for result in r2results],
                 color=colors[15], label='r$^2$', linestyle='--')
ax1.set_ylabel('Correlation \ncoefficient (r$^2$)', color=colors[15], fontsize=8)
ax1.tick_params(axis='y', labelcolor=colors[15], labelsize=8)
ax1.set_ylim(-0.5, 1.1)
ax1.set_yticks([-0.5, 0, 0.5, 1])
line2 = ax2.plot([result[0] for result in diffresults],
                 [result[1] for result in diffresults],
                 color=colors[10], label='RMSE', linestyle='-')
ax2.set_ylabel('Root mean square \nerror (RMSE)', color=colors[10], fontsize=8)
ax2.tick_params(axis='y', labelcolor=colors[10], labelsize=8)
ax2.set_ylim(0, 1.1)
ax2.set_yticks([0, 0.5, 1])
ax2.axvspan(2014 - 0.5, 2014 + 0.5, color=colors[1], alpha=0.2, lw=0)
even_years = [year for year in range(2007, 2023) if year % 2 == 0]
ax1.set_xticks(even_years)
ax1.set_xticklabels([str(year) for year in even_years], fontsize=8, ha='center')
ax2.set_xlim(2006.85, 2022.15)
ax1.annotate(f'r$^2$ = {maxr2:.2f}', xy=(r2year, maxr2),
             xytext=(r2year + 1, maxr2 - 0.1),
             arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle="->"),
             color=colors[15], fontsize=8)
ax2.annotate(f'RMSE = {mindiff:.2f}', xy=(diffyear, mindiff),
             xytext=(diffyear + 1, mindiff - 0.04),
             arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle="->"),
             color=colors[8], fontsize=8)
ax1.yaxis.label.set_size(8)
ax2.yaxis.label.set_size(8)
ax1.xaxis.label.set_size(8)
ax2.xaxis.label.set_size(8)

#  2014
year_data = wind[wind.index.year == 2014]
ax_left_bottom.plot(year_data['yearday'], year_data['speedavg'], label='2014', color=colors[5], alpha=0.7, linewidth=1.5)
ax_left_bottom.plot(mean_yearday.index, mean_yearday, label='Mean', color=colors[16], linewidth=1.5, linestyle='--')
ax_left_bottom.set_xticks([3, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
ax_left_bottom.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], fontsize=8)
ax_left_bottom.set_yticks(np.round(ax_left_bottom.get_yticks(), 1))
ax_left_bottom.set_yticklabels(np.round(ax_left_bottom.get_yticks(), 1), fontsize=8)
ax_left_bottom.set_ylim([1.9, 4.8])
ax_left_bottom.set_xlim([0, 365])
ax_left_bottom.set_ylabel('Wind speed \n(m s$^{-1}$)', fontsize=8)
ax_left_bottom.legend(loc="upper right", fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(1.01, 1.065))

plt.tight_layout(pad=0.8)
plt.show()