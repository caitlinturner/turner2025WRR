import os
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns
import cmocean
from matplotlib.gridspec import GridSpec
from matplotlib.colors import BoundaryNorm
from scipy.stats import gaussian_kde
import pandas as pd

# Load Variables
elevation = np.load('Elevation.npz')
elevation = elevation['elevation']
rows, cols = elevation.shape
num_rows = round(949/30) 
num_cols = round(1790/30)
chunk_size_rows = rows // num_rows
chunk_size_cols = cols // num_cols

## Scenarios ##
#High_Tributary_Open_Diversion
#High_Tributary_Closed_Diversion
#Median_Tributary_Open_Diversion
#Median_Tributary_Closed_Diversion
#Low_Tributary_Open_Diversion
#Low_Tributary_Closed_Diversion


with open('Low_Tributary_Open_Diversion_ExposureData.txt', 'r') as file:
    LExposure = json.load(file)
max_cdf_lake_low_open = 0 
with open('Median_Tributary_Open_Diversion_ExposureData.txt', 'r') as file:
    MExposure = json.load(file)
max_cdf_lake_median_open = 0 
with open('Low_Tributary_Open_Diversion_ExposureData.txt', 'r') as file:
    HExposure = json.load(file)
max_cdf_lake_high_open = 0 


max_clim = 180
timestep_open_closed = 1 
adj = 1

def exposure_time_stats(exp_list, timestep,max_cdf):
    P50 = []  
    p75 = []  
    p90 = []  

    # Iterate through each item in the exposure list
    for item in exp_list:
        # Check if the item is a list with more than 25 unique values
        if isinstance(item, list) and len(set(item)) > 10:
            # Filter out zero values as they represent no exposure
            filtered_list = [x for x in item if x != 0]
            
            # Skip if the filtered list is empty
            if not filtered_list:  
                P50.append(np.nan)
                p75.append(np.nan)
                p90.append(np.nan)
                continue       
            
            # Skip if little variation remains in non-NaN values
            if len(set([x for x in filtered_list if not np.isnan(x)])) < 2:
                P50.append(np.nan)
                p75.append(np.nan)
                p90.append(np.nan)
                continue
            
            # Convert to a NumPy array and remove NaNs
            filtered_array = np.array(filtered_list)
            valid_data = filtered_array[~np.isnan(filtered_array)]
            
            # KDE to approximate the distribution
            kde = gaussian_kde(valid_data)
            x_grid = np.linspace(min(valid_data), max(valid_data), 1000)
            kde_estimates = kde.evaluate(x_grid)
 
            # CDF from the KDE
            cdf = np.cumsum(kde_estimates)
            cdf = cdf / cdf[-1]  
            
            # Account for the max CDF value
            target_p50 = 0.50 + max_cdf
            target_p75 = 0.75 + max_cdf
            target_p90 = 0.90 + max_cdf

            # Find the indices corresponding to the adjusted percentiles
            p50_val_index = np.searchsorted(cdf, target_p50)
            p50_val = x_grid[p50_val_index] if p50_val_index < len(x_grid) else np.nan
            
            p75_val_index = np.searchsorted(cdf, target_p75)
            p75_val = x_grid[p75_val_index] if p75_val_index < len(x_grid) else np.nan

            p90_val_index = np.searchsorted(cdf, target_p90)
            p90_val = x_grid[p90_val_index] if p90_val_index < len(x_grid) else np.nan
            
            # Convert results from units of timesteps to days
            days = 24*adj  # hours in a day
            P50.append(p50_val / timestep / days)  
            p75.append(p75_val / timestep / days)  
            p90.append(p90_val / timestep / days)  
        else:
            # If item doesn't meet criteria, append NaN values
            P50.append(np.nan)
            p75.append(np.nan)
            p90.append(np.nan)

    return P50, p75, p90

def process_exposure(exposure_data, num_rows, num_cols, max_cdf):
    exp_50, exp_75, exp_90 = exposure_time_stats(exposure_data, len(exposure_data), max_cdf)
    exposure_50 = np.array(exp_50).reshape((num_rows, num_cols))
    exposure_75 = np.array(exp_75).reshape((num_rows, num_cols))
    exposure_90 = np.array(exp_90).reshape((num_rows, num_cols))
    return exposure_50, exposure_75, exposure_90

LExposure_50, LExposure_75, LExposure_90 = process_exposure(LExposure, num_rows, num_cols, max_cdf_lake_low_open)
MExposure_50, MExposure_75, MExposure_90 = process_exposure(MExposure, num_rows, num_cols, max_cdf_lake_median_open)
HExposure_50, HExposure_75, HExposure_90 = process_exposure(HExposure, num_rows, num_cols, max_cdf_lake_high_open)

data_50 = [LExposure_50, MExposure_50, HExposure_50]
data_75 = [LExposure_75, MExposure_75, HExposure_75]
data_90 = [LExposure_90, MExposure_90, HExposure_90]


with open('Low_Tributary_Closed_Diversion_ExposureData.txt', 'r') as file:
    LExposure = json.load(file)
with open('Median_Tributary_Closed_Diversion_ExposureData.txt', 'r') as file:
    MExposure = json.load(file)
with open('Low_Tributary_Closed_Diversion_ExposureData.txt', 'r') as file:
    HExposure = json.load(file)
max_cdf_lake_closed = 0 
adj = 4

timestep_open_closed = 4
LExposure_50, LExposure_75, LExposure_90 = process_exposure(LExposure, num_rows, num_cols, max_cdf_lake_closed)
MExposure_50, MExposure_75, MExposure_90 = process_exposure(MExposure, num_rows, num_cols, max_cdf_lake_closed)
HExposure_50, HExposure_75, HExposure_90 = process_exposure(HExposure, num_rows, num_cols, max_cdf_lake_closed)

data_50_closed = [LExposure_50, MExposure_50, HExposure_50]
data_75_closed = [LExposure_75, MExposure_75, HExposure_75]


# Mesh Grid
rows, cols = LExposure_50.shape
scaling_factor_x, scaling_factor_y = 30,30
step_x, step_y = scaling_factor_x, scaling_factor_y
x = np.arange(0, cols * scaling_factor_x, step_x)
y = np.arange(0, rows * scaling_factor_y, step_y)
extent = [0, cols * scaling_factor_x, 0, rows * scaling_factor_y]
XM, YM = np.meshgrid(x, y)

river_coords = {'Blind River': (-10, 345), 'Amite River': (50, 150), 'Tickfaw River': (190, 75),
                'Tangipahoa River': (450, 100), 'Tchefuncte River': (700, -10), 'Pearl River': (1600, 300)}
river_coords_list = list(river_coords.values())
x_river, y_river = zip(*river_coords.values())
spillway_coords = [[225, 750], [240, 690], [320, 580], [370, 620], [325, 675], [275, 715], [275, 750], [225, 750]]
xlims = np.arange(-50, 1800, 100)
ylims = np.arange(-50, 1000, 100)
cbarlims = (0, max_clim)
cbarlevels = 7


windows = 3
def moving_average_filter(data, window_size, threshold=5):
    smoothed_data = np.zeros_like(data)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                smoothed_data[i, j] = np.nan
                continue
        
            i_min = max(i - window_size // 2, 0)
            i_max = min(i + window_size // 2 + 1, data.shape[0])
            j_min = max(j - window_size // 2, 0)
            j_max = min(j + window_size // 2 + 1, data.shape[1])
            
            window = data[i_min:i_max, j_min:j_max]
            window_flat = window.flatten()
            window_flat = window_flat[~np.isnan(window_flat)]
            
            if len(window_flat) == 0:
                smoothed_data[i, j] = np.nan
                continue
            
            median = np.median(window_flat)
            mad = np.median(np.abs(window_flat - median))
            if mad == 0:  
                smoothed_data[i, j] = median
                continue
            
            outlier_mask = np.abs(window_flat - median) > (threshold * mad)
            filtered_window = window_flat[~outlier_mask]
            
            if len(filtered_window) == 0:
                smoothed_data[i, j] = np.nan
            else:
                smoothed_data[i, j] = np.mean(filtered_window)
    
    return smoothed_data

letter_label = ['a', 'd', 'g', 'j',
                'b', 'e', 'h', 'k',
                'c', 'f', 'i', 'l']  
   
def plot_combined_exposure_statistics(data_low, data_medium, data_high):
    fig = plt.figure(figsize=(7.48, 3.4), dpi=600)
    gs = GridSpec(4, 5, figure=fig, hspace=0.06, wspace=0.06, height_ratios=[0.53, 0.53, 0.53, 0.08], width_ratios=[0.05, 1, 1, 1, 1])
    cmap = cmocean.cm.haline_r
    tab10 = sns.color_palette("colorblind", 10)

    scenarios = [("Low", data_low), ("Median", data_medium), ("High", data_high)]
    percentiles = [r'$E_{50}$', r'$E_{50}$', r'$E_{75}$', r'$E_{90}$']

    minval, maxval = cbarlims
    levels = np.linspace(minval, maxval, cbarlevels)

    # Set percentile labels for each column
    for j, percentile in enumerate(percentiles):
        fig.text(0.23 + 0.19 * j, 0.9, percentile, va='center', ha='center', fontsize=8)
        
        letter_index = 0 

    for i, (title, data) in enumerate(scenarios):
        row_title_ax = fig.add_subplot(gs[i, 0])
        row_title_ax.text(0.85, 0.5, title, va='center', ha='center', rotation='vertical', fontsize=8)
        row_title_ax.axis('off')

        for j, single_data in enumerate(data):
            ax = fig.add_subplot(gs[i, j+1])
            
            spillway_polygon = Polygon(spillway_coords, closed=True, fill=True, color=tab10[0], label='Diversion')
            ax.add_patch(spillway_polygon)
            
            ax.scatter(x_river, y_river, color=tab10[2], label='Tributaries', marker='v', s=3)
            
            smoothed_data = moving_average_filter(single_data, window_size=windows)
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
            ax.contourf(XM, YM, smoothed_data, cmap=cmap, norm=norm, algorithm='mpl2014', extend='max')
            
            ax.imshow(elevation, cmap=cmocean.cm.gray, aspect='auto', alpha=0.75)

            ax.set_xlim([xlims[0], xlims[-1]])
            ax.set_ylim([ylims[-1], ylims[0]])
            ax.tick_params(axis='both', which='both', labelsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

            ax.annotate(f'{letter_label[letter_index]}.', xy=(0.02, 0.97), xycoords='axes fraction', fontsize=7.5, fontweight='bold', ha='left', va='top', zorder=100)
            letter_index += 1  

            if i == 0 and j == 0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.74, -0.08), fontsize=6, labelspacing=0.1, frameon=False)

    cbar_left, cbar_bottom, cbar_width, cbar_height = 0.14275, 0.13, 0.758, 0.025
    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    cbar.set_ticks(levels)
    cbar_labels = [str(int(tick)) for tick in levels]
    cbar_labels[-1] = f"{int(levels[-1])}+"
    cbar.ax.set_xticklabels(cbar_labels)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('System-wide exposure time (days)', fontsize=8)
    plt.show()

plot_combined_exposure_statistics(
    [data_50_closed[0], data_50[0], data_75[0], data_90[0]],
    [data_50_closed[1], data_50[1], data_75[1], data_90[1]],
    [data_50_closed[2], data_50[2], data_75[2], data_90[2]])