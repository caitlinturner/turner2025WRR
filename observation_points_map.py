# -*- coding: utf-8 -*-
"""
@author: cturn
"""

import os
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import geopandas as gpd
import numpy as np
import contextily as ctx
import matplotlib.lines as mlines
import cmocean
from shapely.geometry import box, LineString, Polygon
from shapely.ops import unary_union

## Load Data
data_xr_his = xr.open_dataset(r'Calibration\FlowFM_his.nc')
station_names = data_xr_his.station_name.astype(str).to_pandas()
df = data_xr_his.waterlevel.to_pandas()
df.columns = station_names
df.index = df.index - pd.DateOffset(hours=0)

## Main Plot Set Up
fig, ax = plt.subplots(figsize=(7.48, 4.5), dpi=600)
sns.set_style("ticks")
colors = [cmocean.cm.thermal_r(val) for val in np.linspace(0.2, 1, 14)]
louisiana = gpd.read_file('figure_files_data\louisiana.geojson')


## Land Boundaries
trim_poly_coords = [
    (-90.5, 30.0), (-90.0, 30.0), (-89.8, 30.1), (-89.725, 30.15),(-89.75, 30.2), (-90.0, 30.35), #1-5
    (-90.15, 30.38), (-90.16, 30.38), (-90.22, 30.4), 
    (-90.32,30.3), (-90.325,30.28), 
    (-90.425, 30.2), (-90.5, 30.12), (-90.5, 30.0)]
trim_polygon = Polygon(trim_poly_coords)

with open('figure_files_data\finallb1_cutbcs.txt', 'r') as f1:
    coords1 = [tuple(map(float, line.strip().split())) for line in f1]
line1 = LineString(coords1)

with open('figure_files_data\finallb2.txt', 'r') as f2:
    coords2 = [tuple(map(float, line.strip().split())) for line in f2]
line2 = LineString(coords2)

combined_line = unary_union([line1, line2])
trimmed_line = combined_line.intersection(trim_polygon)


ax.plot(*line1.xy, color=colors[12], linewidth=1.5)
ax.plot(*line2.xy, color=colors[12], linewidth=1.5)

if not trimmed_line.is_empty:
    if trimmed_line.geom_type == 'LineString':
        ax.plot(*trimmed_line.xy, color=colors[3], linewidth=1.5, linestyle='--')
    elif trimmed_line.geom_type == 'MultiLineString':
        for segment in trimmed_line.geoms:
            ax.plot(*segment.xy, color=colors[3], linewidth=1.5, linestyle='--')

lb_full = mlines.Line2D([], [], color=colors[12], linewidth=2, label=r'Model Boundary (Delft3DFM + $\mathit{dorado}$)')
lb_trim = mlines.Line2D([], [], color=colors[3], linewidth=2, linestyle='--', label=r'Analysis Boundary ($\mathit{dorado}$)')
land_legend = ax.legend(handles=[lb_full, lb_trim], loc='lower right', frameon=True, fontsize=7)
ax.add_artist(land_legend)


## Observation Stations
stations = [
    {'name': 'The Rigolets (USGS)',     'x': -89.7167, 'y': 30.1578},
    {'name': 'Chef Menteur (USACE)',    'x': -89.8021, 'y': 30.0661},
    {'name': 'Mandeville (USACE)',      'x': -90.0939, 'y': 30.3596},
    {'name': 'Shell Beach (NOAA)',      'x': -90.3986, 'y': 30.2862},
    {'name': 'Pass Manchac (USACE)',    'x': -89.672,  'y': 29.8926},
    {'name': 'New Canal (NOAA)',        'x': -90.1133, 'y': 30.0267}, 
    {'name': 'Pearl River (USGS)',      'x': -89.585,  'y': 30.1653},
    {'name': 'Causeway (USGS)',         'x': -90.1227, 'y': 30.1996},
    {'name': 'The Rigolets (LSU)',      'x': -89.6955, 'y': 30.1696},
    {'name': 'Maurepas (LSU)',          'x': -90.3107, 'y': 30.279803},
    {'name': 'BCS (LSU)',               'x': -90.3755, 'y': 30.09551},
    {'name': 'North Shore (LSU)',       'x': -90.2121, 'y': 30.3668},]

index_values = [0,1,2,3,4,5,6,7,8,9,10,11] 
markers = ['o', 's', '^', 'D', 'v', 'p', 'P', '*']
num_stations = len(stations)
repeated_markers = (markers * ((num_stations // len(markers)) + 1))[:num_stations]
station_names = data_xr_his['station_name'].astype(str).values[index_values]
legend_handles = []

for i, idx in enumerate(index_values):
    station = stations[idx]
    if station['x'] is not None and station['y'] is not None:
        ax.scatter(station['x'], station['y'], marker=repeated_markers[i], color=colors[i],
                   s=35, edgecolor='black', zorder=3)
    handle = mlines.Line2D([], [], color=colors[i], marker=repeated_markers[i], linestyle='None',
                           markersize=6, label=station['name'], markeredgecolor='black')
    legend_handles.append(handle)

ax.legend(handles=legend_handles, loc='lower left',
          bbox_to_anchor=(0.001, 0.001, 1, 1), ncol=2, fontsize=8,
          title="Observation Stations", title_fontsize=8, labelspacing=0.3)


## Major Locations
label_locations = {
    ' Lake \n Borgne': (-89.636, 30.027),
    '  Lake \n Maurepas': (-90.522, 30.23),
    'Bonnet Carré \n Spillway': (-90.335, 29.96)
}
for label, coords in label_locations.items():
    ax.annotate(label, xy=coords, xytext=(5, 5), textcoords="offset points", ha='center', fontsize=8)

label_locations = {'Lake Pontchartrain': (-90.15, 30.205)}
for label, coords in label_locations.items():
    ax.annotate(label, xy=coords, xytext=(5, 5), textcoords="offset points", ha='center', fontsize=12)
    

## BCS Polygon
bcs_x = [-90.455, -90.452, -90.405, -90.373, -90.394, -90.423, -90.429, -90.455]
bcs_y = [30.0, 30.0195, 30.078, 30.056, 30.029, 30.010, 30.0, 30.0]
ax.fill(bcs_x, bcs_y, color=colors[1], alpha=1, zorder=3)
    
    
## Channels and Stations
channel_names = ['R. Tidal Pass', '   C. M. \nTidal Pass', 'Tidal Pass']
channel_coords = [(-89.7, 30.175), (-89.889, 30.04),(-90.375, 30.29)]
for name, coord in zip(channel_names, channel_coords):
    ax.annotate(name, xy=coord, xytext=(5, 5), textcoords="offset points", ha='center', fontsize=7)
    
## Boundary Names
boundary_names = {'Meteorological\nSite': 'Meteorological\nSite', 'x': -90.0341, 'y': 30.0301}
boundary_coords = [(-90.12, 29.958)]    
for name, coord in zip(boundary_names, boundary_coords):
    ax.annotate(name, xy=coord, xytext=(5, 5), textcoords="offset points", ha='center', fontsize=7)


## Rivers
river_names = ['Mississippi R.', 'Blind R.', 'Amite R.', 'Tickfaw R.', 'Tangipahoa R.', 'Tchefuncte R.', 'Pearl R.']
river_coords = [(-90.608, 29.99), (-90.65, 30.21), (-90.635, 30.29), (-90.527, 30.34), (-90.36, 30.34), (-90.125, 30.3825), (-89.6, 30.2)]
for name, coord in zip(river_names, river_coords):
    ax.annotate(name, xy=coord, xytext=(5, 5), textcoords="offset points", ha='center', fontsize=8)

ax.annotate('Gulf \n of \n Mexico', xy=(-89.442, 30.0825), xytext=(5, 5), textcoords="offset points", ha='center', fontsize=9)


# === Configure Map ===
ax.set_xlim([-90.71, -89.37])
ax.set_ylim([29.7, 30.5])
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.VoyagerNoLabels, attribution=False)

# === Inset Map ===
ax_inset = inset_axes(ax, width="29%", height="29%", loc='upper right',
                      bbox_to_anchor=[0, 0, 1.069, 1.012], bbox_transform=ax.transAxes)

for spine in ax_inset.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1)

louisiana.boundary.plot(ax=ax_inset, color='black', linewidth=0.5)
ax_inset.plot(*line1.xy, color=colors[12], linewidth=1.5)
ax_inset.plot(*line2.xy, color=colors[12], linewidth=1.5)

river_line = gpd.read_file('figure_files_data\river_line.shp')
y_lims = [28.146, 32.80]
bbox = box(minx=-180, maxx=180, miny=y_lims[0], maxy=y_lims[1])
filtered_rivers = river_line[river_line.intersects(bbox)]
for river_name in ['Mississippi River', 'Red River']:
    river = filtered_rivers[filtered_rivers['name'] == river_name]
    river.plot(ax=ax_inset, color='steelblue', linewidth=0.75, zorder=3)

ax_inset.set_xticks([])
ax_inset.set_yticks([])
ax_inset.set_ylim([y_lims[0] + 0.5, y_lims[1] + 0.5])
ax_inset.annotate('N', xy=(0.85, 0.98), xytext=(0.85, 0.78),
                  arrowprops=dict(facecolor='black', width=5, headwidth=8, headlength=10),
                  ha='center', va='top', fontsize=10, xycoords='axes fraction')
