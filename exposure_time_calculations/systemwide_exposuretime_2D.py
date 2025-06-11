# -*- coding: utf-8 -*-
"""
@author: cturn65
"""
import os
import numpy as np
import json
import time as tm
import dorado.particle_track as pt  
starttm = tm.time()

# Set File and Folder Names
scenario_name = 'High_Tributary_Open_Diversion'

## Scenarios ##
#High_Tributary_Open_Diversion
#High_Tributary_Closed_Diversion
#Median_Tributary_Open_Diversion
#Median_Tributary_Closed_Diversion
#Low_Tributary_Open_Diversion
#Low_Tributary_Closed_Diversion


# Load Data
with open(f'{scenario_name}_WalkData.txt', 'r') as file:
    walk_data = json.load(file)

elevation = np.load('elevation.npz')['elevation']
celltype_lponly = np.genfromtxt(r"exposure_time_calculations\required_files\gridcell_type_lp_isolated.csv", delimiter=',')


walk_data = {}

# Load data for the single scenario
with open(f'Exposure_Time_Calculations\{scenario_name}_WalkData.txt', 'r') as file:
    walk_data[scenario_name] = json.load(file)
walk_data = walk_data[f'{scenario_name}']



elevation = np.load('elevation.npz')['elevation']
celltype_lponly = np.genfromtxt("gridcell_type_lp_only.csv", delimiter=',')

# Identify Cells for Lake Pontchartrain Only
is_lake = celltype_lponly == 1
seed_xlocexp, seed_ylocexp = np.where(is_lake)
regions = np.zeros_like(elevation, dtype='int')
regions[seed_xlocexp, seed_ylocexp] = 1


# Set Chunk Dimensions
rows, cols = regions.shape
num_rows = round(rows / 30) 
num_cols = round(cols / 30)
chunk_size_rows = rows // num_rows
chunk_size_cols = cols // num_cols


def find_common_indices_info(data, x_key, y_key, xind_range, yind_range):
    if x_key not in data or y_key not in data:
        return {}

    common_info = {key: [] for key in data.keys()}

    xinds = data[x_key]
    yinds = data[y_key]

    if not xinds or not yinds:
        return common_info

    xind_min, xind_max = xind_range
    yind_min, yind_max = yind_range

    count = 0 

    for index in range(len(xinds)):
        if xinds[index] and yinds[index] and (xind_min <= xinds[index][0] <= xind_max) and (yind_min <= yinds[index][0] <= yind_max):
            for key in data.keys():
                common_info[key].append(data[key][index])
            count += 1

    return common_info



# Process Exposure Time for Each Chunk
exposure_time_data_chunks = []
for chunk_row in range(num_rows):
    for chunk_col in range(num_cols):
        start_row = chunk_row * chunk_size_rows
        end_row = min((chunk_row + 1) * chunk_size_rows, rows)
        start_col = chunk_col * chunk_size_cols
        end_col = min((chunk_col + 1) * chunk_size_cols, cols)
        
        if np.any(is_lake[start_row:end_row, start_col:end_col]):       
            chunk_walk_data = find_common_indices_info(walk_data, 'xinds', 'yinds',(start_row,end_row),(start_col,end_col))
            exposure_time_chunk_data = pt.exposure_time(chunk_walk_data, regions, verbose=False)
            exposure_time_data_chunks.append(exposure_time_chunk_data)
        else:
            exposure_time_data_chunks.append(np.nan)
            
     
json.dump(exposure_time_data_chunks, open(f'{scenario_name}_ExposureData_system-wide.txt', 'w'))