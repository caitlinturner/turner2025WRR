# -*- coding: utf-8 -*xdsz
"""

@author: cturn
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import cmocean
import json
import time as tm
import scipy
import scipy.interpolate
import scipy.ndimage
from scipy.stats import kruskal, mannwhitneyu
from itertools import combinations
import pandas as pd


# Start Timer
starttm = tm.time()

# Set Directories and Load Variables
data_dir = r'Exposure_Time_Calculations'
os.chdir(data_dir)

# Load exposure map
required_vars = np.load('Elevation.npz')
elevation = required_vars['elevation']
timedelta = 86400  ## Go into source code if error pops up and change '==' to '=' for day.
timeunit = '[day]'

# Function to load JSON data
def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Load and process cell type data
celltype_lponly = np.loadtxt(r'Exposure_Time_Calculations\Required_Files\gridcell_type_lp_only.csv', delimiter=',')
is_lake = celltype_lponly == 1
seed_xlocexp, seed_ylocexp = np.where(is_lake)
regions = np.zeros_like(elevation, dtype='int')
regions[seed_xlocexp, seed_ylocexp] = 1

# Load exposure times
def load_exposure_times(filenames):
    return [np.array(load_json(filename)) for filename in filenames]

filenames = [
    'Exposure_Time_Calculations/High_Tributary_Open_Diversion/figs/data/exposure_times.txt',
    'Exposure_Time_Calculations/High_Tributary_Closed_Diversion/figs/data/exposure_times.txt',
    'Exposure_Time_Calculations/Median_Tributary_Open_Diversion/figs/data/exposure_times.txt',
    'Exposure_Time_Calculations/Median_Tributary_Closed_Diversion/figs/data/exposure_times.txt',
    'Exposure_Time_Calculations/Low_Tributary_Open_Diversion/figs/data/exposure_times.txt',
    'Exposure_Time_Calculations/Low_Tributary_Closed_Diversion/figs/data/exposure_times.txt']

L_exposure_times, M_exposure_times, H_exposure_times, CL_exposure_times, CM_exposure_times, CH_exposure_times = load_exposure_times(filenames)

def plot_smooth_etd(exposure_times, Np_tracer, common_end_time, label, color, ax, nbins, timedelta, linestyle, sigma=1):
    plotting_times = exposure_times[exposure_times > 1e-6]
    plotting_times = plotting_times[plotting_times < 0.99 * common_end_time]
    num_particles_included = len(plotting_times)
    full_time_vect = np.append([0], np.sort(plotting_times))
    full_time_vect = np.append(full_time_vect, [common_end_time])
    frac_exited = np.arange(0, num_particles_included + 1, dtype='float') / Np_tracer
    frac_exited = np.append(frac_exited, [float(num_particles_included) / float(Np_tracer)])
    create_linear_CDF = scipy.interpolate.interp1d(full_time_vect, frac_exited, kind='linear')
    smooth_time_vect = np.linspace(0, common_end_time, nbins)
    linear_CDF = create_linear_CDF(smooth_time_vect)
    timestep = smooth_time_vect[1] - smooth_time_vect[0]
    RTD = np.gradient(linear_CDF, timestep)
    RTD_smooth = scipy.ndimage.gaussian_filter1d(RTD, sigma=sigma)
    ax.plot(smooth_time_vect / timedelta, RTD * timedelta, label=label, color=color, linewidth=1.5, linestyle=linestyle)

def plot_cdf(exposure_times, Np_tracer, common_end_time, label, color, ax, nbins, timedelta, linestyle):
    plotting_times = exposure_times[exposure_times > 1e-6]
    plotting_times = plotting_times[plotting_times < 0.99 * common_end_time]
    num_particles_included = len(plotting_times)
    full_time_vect = np.append([0], np.sort(plotting_times))
    full_time_vect = np.append(full_time_vect, [common_end_time])
    frac_exited = np.arange(0, num_particles_included + 1, dtype='float') / Np_tracer
    frac_exited = np.append(frac_exited, [float(num_particles_included) / float(Np_tracer)])
    create_smooth_CDF = scipy.interpolate.interp1d(full_time_vect, frac_exited, kind='previous')
    smooth_time_vect = np.linspace(0, common_end_time, nbins)
    smooth_CDF = create_smooth_CDF(smooth_time_vect)
    ax.plot(smooth_time_vect / timedelta, smooth_CDF, label=label, color=color, linewidth=1.5, linestyle=linestyle)
    return smooth_CDF

def calc_cdf(exposure_times, Np_tracer, common_end_time, nbins):
    plotting_times = exposure_times[exposure_times > 1e-6]
    plotting_times = plotting_times[plotting_times < 0.99 * common_end_time]
    num_particles_included = len(plotting_times)
    full_time_vect = np.append([0], np.sort(plotting_times))
    full_time_vect = np.append(full_time_vect, [common_end_time])
    frac_exited = np.arange(0, num_particles_included + 1, dtype='float') / Np_tracer
    frac_exited = np.append(frac_exited, [float(num_particles_included) / float(Np_tracer)])
    create_smooth_CDF = scipy.interpolate.interp1d(full_time_vect, frac_exited, kind='previous')
    smooth_time_vect = np.linspace(0, common_end_time, nbins)
    smooth_CDF = create_smooth_CDF(smooth_time_vect)
    smooth_CDF_y = smooth_CDF
    smooth_CDF_x = smooth_time_vect / timedelta
    return smooth_CDF_x, smooth_CDF_y

def find_closest_x_values(x, y, target_y_values):
    closest_x_values = {}
    for target_y in target_y_values:
        idx = (np.abs(y - target_y)).argmin()
        closest_x_values[target_y] = round(x[idx], 1)
    return closest_x_values


# Open
nbins_open = 30
max_open_val = 80
common_end_time_spillway = 86400 * max_open_val

# Closed
nbins_closed = 60
max_open_closed = 636
common_end_time_closed = 86400 * max_open_closed


target_y_values = [0.5, 0.75, 0.9]

L_smooth_CDF_x, L_smooth_CDF_y = calc_cdf(L_exposure_times, len(L_exposure_times), common_end_time_spillway, nbins_open)
M_smooth_CDF_x, M_smooth_CDF_y = calc_cdf(M_exposure_times, len(M_exposure_times), common_end_time_spillway, nbins_open)
H_smooth_CDF_x, H_smooth_CDF_y = calc_cdf(H_exposure_times, len(H_exposure_times), common_end_time_spillway, nbins_open)
L_closest_x_values = find_closest_x_values(L_smooth_CDF_x, L_smooth_CDF_y, target_y_values)
M_closest_x_values = find_closest_x_values(M_smooth_CDF_x, M_smooth_CDF_y, target_y_values)
H_closest_x_values = find_closest_x_values(H_smooth_CDF_x, H_smooth_CDF_y, target_y_values)
print("L CDF closest x values:", L_closest_x_values)
print("M CDF closest x values:", M_closest_x_values)
print("H CDF closest x values:", H_closest_x_values)


CL_smooth_CDF_x, CL_smooth_CDF_y = calc_cdf(CL_exposure_times, len(CL_exposure_times), common_end_time_closed, nbins_closed)
CM_smooth_CDF_x, CM_smooth_CDF_y = calc_cdf(CM_exposure_times, len(CM_exposure_times), common_end_time_closed, nbins_closed)
CH_smooth_CDF_x, CH_smooth_CDF_y = calc_cdf(CH_exposure_times, len(CH_exposure_times), common_end_time_closed, nbins_closed)
CL_closest_x_values = find_closest_x_values(CL_smooth_CDF_x, CL_smooth_CDF_y, target_y_values)
CM_closest_x_values = find_closest_x_values(CM_smooth_CDF_x, CM_smooth_CDF_y, target_y_values)
CH_closest_x_values = find_closest_x_values(CH_smooth_CDF_x, CH_smooth_CDF_y, target_y_values)
print("CL CDF closest x values:", CL_closest_x_values)
print("CM CDF closest x values:", CM_closest_x_values)
print("CH CDF closest x values:", CH_closest_x_values)


# Set up the plot
fig, axes = plt.subplots(2, 2, figsize=(4.48, 4), dpi=600, sharex='col', sharey='row')
linestyles = ['--', '-', ':', '-.', 'solid']
colors = [cmocean.cm.thermal_r(0.2), cmocean.cm.thermal_r(0.5), cmocean.cm.thermal_r(0.9), cmocean.cm.haline(0.2), cmocean.cm.haline(0.9)]

# Spillway Open
# ETD Plot
plot_smooth_etd(H_exposure_times, len(H_exposure_times), common_end_time_spillway, 'High', colors[1], axes[0, 0], nbins_open, timedelta, linestyles[0])
plot_smooth_etd(M_exposure_times, len(M_exposure_times), common_end_time_spillway, 'Median', colors[2], axes[0, 0], nbins_open, timedelta, linestyles[1])
plot_smooth_etd(L_exposure_times, len(L_exposure_times), common_end_time_spillway, 'Low', colors[0], axes[0, 0], nbins_open, timedelta, linestyles[2])
axes[0, 0].set_ylabel('Probability density \n function (${\\text{d}^{-1}}$)', fontsize=8)
axes[0, 0].set_xlim([0, max_open_val])
axes[0, 0].set_ylim([0, 0.065])
axes[0, 0].tick_params(axis='x', labelsize=8)
axes[0, 0].tick_params(axis='y', labelsize=8)
axes[0, 0].set_yticks([0.0, 0.02, 0.04, 0.06])

# CDF Plot
plot_cdf(H_exposure_times, len(H_exposure_times), common_end_time_spillway, 'High', colors[1], axes[1, 0], nbins_open, timedelta, linestyles[0])
plot_cdf(M_exposure_times, len(M_exposure_times), common_end_time_spillway, 'Median', colors[2], axes[1, 0], nbins_open, timedelta, linestyles[1])
plot_cdf(L_exposure_times, len(L_exposure_times), common_end_time_spillway, 'Low', colors[0], axes[1, 0], nbins_open, timedelta, linestyles[2])
axes[1, 0].set_ylabel('Cumulative \n distribution function [-]', fontsize=8)
axes[1, 0].set_xlim([0, max_open_val])
axes[1, 0].set_ylim([0, 1.02])
axes[1, 0].tick_params(axis='x', labelsize=8)
axes[1, 0].tick_params(axis='y', labelsize=8)
axes[1, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

right_ax1 = axes[1, 0].twinx()
right_ax1.set_ylim([0, 1.02])
right_ax1.set_yticks([0.50, 0.75, 0.90])
right_ax1.set_yticklabels('')
right_ax1.tick_params(axis='y', labelsize=8)
right_ax1.grid(True, which='both', axis='y', alpha=0.75)

# Spillway Closed
# ETD Plot
plot_smooth_etd(CH_exposure_times, len(CH_exposure_times), common_end_time_closed, 'High', colors[1], axes[0, 1], nbins_closed, timedelta, linestyles[0])
plot_smooth_etd(CM_exposure_times, len(CM_exposure_times), common_end_time_closed, 'Median', colors[2], axes[0, 1], nbins_closed, timedelta, linestyles[1])
plot_smooth_etd(CL_exposure_times, len(CL_exposure_times), common_end_time_closed, 'Low', colors[0], axes[0, 1], nbins_closed, timedelta, linestyles[2])
axes[0, 1].legend(fontsize=8, frameon=False, labelspacing=0.2)
axes[0, 1].set_xlim([0, max_open_closed])
axes[0, 1].set_ylim([0, 0.065])
axes[0, 1].tick_params(axis='x', labelsize=8)
axes[0, 1].tick_params(axis='y', labelsize=8)
#axes[0, 1].set_title('Diversion closed', fontsize=8, pad=3)

# CDF Plot
plot_cdf(CH_exposure_times, len(CH_exposure_times), common_end_time_closed, 'High', colors[1], axes[1, 1], nbins_closed, timedelta, linestyles[0])
plot_cdf(CM_exposure_times, len(CM_exposure_times), common_end_time_closed, 'Median', colors[2], axes[1, 1], nbins_closed, timedelta, linestyles[1])
plot_cdf(CL_exposure_times, len(CL_exposure_times), common_end_time_closed, 'Low', colors[0], axes[1, 1], nbins_closed, timedelta, linestyles[2])
axes[1, 1].set_xlim([0, max_open_closed])
axes[1, 1].set_ylim([0, 1.02])
axes[1, 1].tick_params(axis='x', labelsize=8)
axes[1, 1].tick_params(axis='y', labelsize=8)
axes[1, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

right_ax2 = axes[1, 1].twinx()
right_ax2.set_ylim([0, 1.02])
right_ax2.set_yticks([0.50, 0.75, 0.90])
right_ax2.set_yticklabels([r'$E_{50}$', r'$E_{75}$', r'$E_{90}$'])
right_ax2.tick_params(axis='y', labelsize=8)
right_ax2.grid(True, which='both', axis='y', alpha=0.75)


fig.text(0.5, 0.03, 'Time (days)', ha='center', va='center', fontsize=8)
labels = ['a.', 'c.', 'b.', 'd.']
for i, ax in enumerate(axes.flat):
    ax.annotate(labels[i], xy=(0.02, 0.985), xycoords='axes fraction', fontsize=8, fontweight='bold', ha='left', va='top', zorder=100)

plt.subplots_adjust(hspace=0.2, wspace=0.2)
plt.show()



### Stats ###
# Combine data into groups for open and closed scenarios
open_scenarios = {
    'Low': L_smooth_CDF_y,
    'Median': M_smooth_CDF_y,
    'High': H_smooth_CDF_y
}

closed_scenarios = {
    'Low': CL_smooth_CDF_y,
    'Median': CM_smooth_CDF_y,
    'High': CH_smooth_CDF_y
}

# Function to perform Kruskal-Wallis H-test and post-hoc pairwise comparisons
def stats_test(scenarios, group_name):
    # Perform Kruskal-Wallis H-test
    kruskal_stat, kruskal_p = kruskal(*scenarios.values())
    print(f"{group_name} Scenarios - Kruskal-Wallis H-test: H = {kruskal_stat:.3f}, p = {kruskal_p:.3f}")
    
    # Pairwise Mann-Whitney U-tests
    scenario_pairs = list(combinations(scenarios.keys(), 2))
    pairwise_results = []
    
    for scenario1, scenario2 in scenario_pairs:
        group1 = scenarios[scenario1]
        group2 = scenarios[scenario2]
        U, p = mannwhitneyu(group1, group2, alternative='two-sided')
        pairwise_results.append((scenario1, scenario2, U, p))
    
    # Apply Bonferroni correction
    bonferroni_alpha = 0.05 / len(pairwise_results)
    pairwise_results_corrected = [
        (s1, s2, U, p, p < bonferroni_alpha) for s1, s2, U, p in pairwise_results
    ]
    
    # Create a DataFrame for pairwise results
    pairwise_df = pd.DataFrame(pairwise_results_corrected, 
                                columns=['Scenario 1', 'Scenario 2', 'U Statistic', 'p-value', 'Significant (Bonferroni)'])
    print(f"{group_name} Scenarios Pairwise Comparisons:")
    print(pairwise_df)
    return pairwise_df

# Analyze open and closed scenarios
open_results = stats_test(open_scenarios, "Open")
closed_results = stats_test(closed_scenarios, "Closed")