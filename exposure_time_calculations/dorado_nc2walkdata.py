# -*- coding: utf-8 -*-
"""
@author: cturn
"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xarray as xr
import cmocean
import json
import time as tm
import dorado
import dorado.particle_track as pt
import warnings
warnings.filterwarnings('ignore')
starttm = tm.time()

 
## Set File and Folder Names
scenario_name = 'High_Tributary_Open_Diversion'
file_name = 'High_Tributary_Open_Diversion'

## Scenarios ##
#High_Tributary_Open_Diversion
#High_Tributary_Closed_Diversion
#Median_Tributary_Open_Diversion
#Median_Tributary_Closed_Diversion
#Low_Tributary_Open_Diversion
#Low_Tributary_Closed_Diversion


## Set Times
startRange =  2808                               # Start time of interest by hour (Model provided hourly timesteps)
endRange = 6553                                # End time of interest by hour # Sep 1 (6553) June 1 (4345) Full Time (18261)
hydrodt = 2                                     # Time step of interest (hours)
timesteps = (endRange - startRange) // hydrodt  # Total timesteps for Dorado run
model_timestep = 3600 * hydrodt                 # Timsteps between data in seconds

# Set Number of Particles
particles = 20000

## Set Directories Up ## -----------------------------------------------------
path2folder = scenario_name
os.makedirs(path2folder, exist_ok=True)

## Load Data ## --------------------------------------------------------------

celltype_lponly = np.genfromtxt(r"exposure_time_calculations\required_files\gridcell_type.csv", delimiter=',')
boundaries = pd.read_excel(r"exposure_time_calculations\required_files\boundaries_latlon.xlsx")

dir_testinput = os.path.join(f'hydroshare_data/Hydrodynamic_Scenarios/{scenario_name}/{scenario_name}.dsproj_data/FlowFM/output')
file_nc = os.path.join(dir_testinput, 'FlowFM_map.nc')
map_xr = xr.open_dataset(file_nc)

elevation = map_xr['mesh2d_flowelem_bl'].values.astype(np.float32)
map_xr = map_xr.rename({})
nc_varkeys = map_xr.variables.mapping.keys() 
list_varattrs_pd = []
for varkey in nc_varkeys:
    varattrs_pd = pd.DataFrame({varkey:map_xr.variables.mapping[varkey].attrs}).T
    varattrs_pd[['shape','dimensions','dtype']] = 3*[''] 
    varattrs_pd.at[varkey,'shape'] = map_xr[varkey].shape
    varattrs_pd.at[varkey,'dimensions'] = map_xr.variables[varkey].dims
    varattrs_pd.at[varkey,'dtype'] = map_xr.variables[varkey].dtype
    list_varattrs_pd.append(varattrs_pd)
vars_pd = pd.concat(list_varattrs_pd,axis=0)
vars_pd[vars_pd.isnull()] = ''
vars_pd = vars_pd[["standard_name","units","location","shape","dimensions","dtype","long_name"]]


## Change Decimal Degree to UTM Projection ## --------------------------------
def decimal_degrees_to_utm(lat, lon):
    from pyproj import Proj
    utm_proj = Proj(proj='utm', zone=15, ellps='WGS84')
    return utm_proj(lon, lat)
x, y = decimal_degrees_to_utm(map_xr['mesh2d_face_y'].values, map_xr['mesh2d_face_x'].values)
boundaries.x, boundaries.y = decimal_degrees_to_utm(boundaries.y, boundaries.x)



## Create Rasterization Function for Hydrodynamic Flexible Mesh Grid ## -----------------------------
resolution, knn = 60, 10
myInterp, elevation = pt.unstruct2grid(list(zip(x.tolist(), y.tolist())), elevation.tolist(), resolution, knn, boundary=boundaries, crop=True)

# Set ocean boundary and fix area that was cut
interpolate_indices = (533, 544, 1296, 1304)
ocean_boundary = (394, 551, 1788, 1790)
elevation[interpolate_indices[0]:interpolate_indices[1], interpolate_indices[2]:interpolate_indices[3]] = pd.DataFrame(elevation[interpolate_indices[0]:interpolate_indices[1], interpolate_indices[2]:interpolate_indices[3]]).interpolate().values

np.savez_compressed('Elevation.npz', elevation=elevation) # Only needs to be saved once

# Initialize cell types
# Spillway Open
celltype_initial = np.where(~np.isnan(elevation), 1, np.nan) 
# Spillway Closed
# celltype_initial = np.genfromtxt(r'exposure_time_calculations\required_files\gridcell_type_lp_isolated.csv', delimiter=',')


## Set Particles ## ----------------------------------------------------------
regions = np.zeros_like(elevation, dtype='int')
seed_locations = [(i, j) for i in range(celltype_lponly.shape[0]) for j in range(celltype_lponly.shape[1]) if celltype_lponly[i, j] == 1]
rows, cols = celltype_lponly.shape
for i in range(rows):
    for j in range(cols):
        if celltype_lponly[i, j] == 1:
            regions[i, j] = 1  
np.random.shuffle(seed_locations)
num_seeds = int(len(seed_locations) * 0.05)
selected_seed_locations = seed_locations[:num_seeds]
seed_xloc, seed_yloc = zip(*selected_seed_locations)

# Paritcle Initial Locations with new indicies
fig, ax = plt.subplots(figsize=(10, 5), dpi=400)
cax = ax.imshow(elevation, cmap=cmocean.cm.deep_r, vmin=-25, vmax=0)
plt.imshow(regions, cmap='bone', alpha=0.3)
plt.scatter(seed_yloc,seed_xloc, s = 0.5, color='darkblue')
cbar = fig.colorbar(cax, ax=ax, label='Depth (m)',shrink = 0.88)
ax.set_xlabel('Longitude Index', fontsize=12)
ax.set_ylabel('Latitude Index', fontsize=12)
cbar.ax.set_ylabel('Depth (m)', fontsize=12)
cbar.set_ticks([0, -5, -10, -15, -20, -25])
cbar.set_ticklabels(['0', '5', '10','15', '20','25'])
ax.set_facecolor('whitesmoke')
plt.show()

# Set Dorado parameters that remain static
walk_data = {}
particles = 5000
model_timestep = 3600  # seconds (1 hour)
timesteps = (len(unstructured['time'])-1)

params = pt.modelParams()
params.theta = 1.0
params.gamma = 0.05
params.diff_coeff = 0.8
params.cell_type = np.where(np.isnan(elevation), 2, celltype_initial)
params.dx = 100
params.dry_depth = 0.01
target_times = np.arange(0, model_timestep * (timesteps + 1), model_timestep)
target_times = [int(t) for t in target_times]
params.verbose = True
prev_counts = None # Initialize for particle timing

for i in range(timesteps):
    # Rasterize all data
    stage = myInterp(unstructured['timesteps'][i][0])
    depth = myInterp(unstructured['timesteps'][i][1])
    u = myInterp(unstructured['timesteps'][i][2])
    v = myInterp(unstructured['timesteps'][i][3])

    for arr in [u, v]:
        arr[ocean_boundary_r[0]:ocean_boundary_r[1], ocean_boundary_r[2]:ocean_boundary_r[3]] = 0
        arr[ocean_boundary_cm[0]:ocean_boundary_cm[1], ocean_boundary_cm[2]:ocean_boundary_cm[3]] = 0
        
    # Set Dorado parameters
    params.topography = elevation
    params.stage = stage
    params.qx = u * depth
    params.qy = v * depth
    params.u = u
    params.y = v

    # Generate particles
    particle = pt.Particles(params)
    if i == 0:
        particle.generate_particles(particles, seed_xloc, seed_yloc, seed_time=0, method='exact')
    else:
        particle.generate_particles(0, xi, yi, seed_time=0, method='exact', previous_walk_data=walk_data)

    # Run iteration
    walk_data = particle.run_iteration(target_times[i])
    xi, yi, ti = dorado.routines.get_state(walk_data)

    ## Account for travel_times of particles
    x_key = 'xinds' if 'xinds' in walk_data else 'x_inds'
    t_key = 'travel_times'
    x_lists = walk_data[x_key]
    t_lists = walk_data.get(t_key, [[] for _ in x_lists])
    walk_data[t_key] = t_lists
    
    if prev_counts is None or len(prev_counts) != len(x_lists):
        prev_counts = [0] * len(x_lists)
    t_start = target_times[i - 1] if i > 0 else 0
    t_end   = target_times[i]
    for p, x_list in enumerate(x_lists):
        cur_len = len(x_list)
        added = cur_len - prev_counts[p]
        if added <= 0:
            continue
    
        # Determine starting time (last known time if any) and build travel_times list
        last_time = t_lists[p][prev_counts[p]-1] if prev_counts[p] > 0 else t_start
        seg = np.linspace(last_time, t_end, added + 1, endpoint=True)[1:]
        seg = [int(s) if s.is_integer() else float(s) for s in seg]
     
        # Increase travel_times array if multiple iterations are needed
        if len(t_lists[p]) < cur_len:
            t_lists[p].extend([None] * (cur_len - len(t_lists[p])))
        t_lists[p][prev_counts[p]:cur_len] = seg
        prev_counts[p] = cur_len


    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ax.scatter(yi, xi, c='firebrick', s=1)
    im = ax.imshow(particle.depth * -1, vmin=0, vmax=-25)
    plt.title(f'Time: {int((target_times[i] / 3600) // 24)} Days, {int((target_times[i] / 3600) % 24)} Hours')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Depth (m)')
    cbar.set_ticks([0, -5, -10, -15, -20, -25])
    cbar.set_ticklabels(['0', '5', '10', '15', '20', '25'])
    im.axes.xaxis.set_ticklabels([])
    im.axes.yaxis.set_ticklabels([])
    im.axes.get_xaxis().set_ticks([])
    im.axes.get_yaxis().set_ticks([])
    ax.set_facecolor('whitesmoke')
    plt.savefig(f'{path2folder}/output_by_dt{i}.png')
    plt.close()

## Exposure Time Calculation ## ----------------------------------------------
exposure_times = pt.exposure_time(walk_data, regions)
exposure_times_plot = dorado.routines.plot_exposure_time(walk_data, exposure_times, f'{scenario_name}/figs', timedelta=86400, nbins=20)


## Save Results ## ----------------------------------------------------------
json.dump(walk_data, open(f'{scenario_name}_WalkData.txt', 'w'))
np.savez_compressed(f'{scenario_name}_Exposure.npz', exposure_times=exposure_times, elevation=elevation, celltype_lponly=celltype_lponly, model_timestep=model_timestep)


## Get Runtime ## -----------------------------------------------------------
elapsed = (tm.time() - starttm) / 60
hrs, mins = divmod(elapsed, 60)
print(f"Time Taken: {int(hrs)} hours, {round(mins)} minutes")




# import ffmpeg
# import matplotlib.animation as animation
# from dorado.routines import animate_plots

# animate_plots(0, 1536, f'exposure_time_calculations/Final_Versions/{Scenario_Name}')















