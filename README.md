# turner2025WRR
This repository (turner2025WRR) supports the hydrodynamic modeling study described in "Water exposure time distributions controlled by freshwater releases in a semi-enclosed estuary," published in Water Resources Research by Turner and Hiatt (2025). The model was developed using the Delft3D Flexible Mesh (D3DFM) system to simulate circulation dynamics in the Lake Pontchartrain Estuary (LPE). Model input files to reproduce the results of this code can be found on Hydroshare (Turner & Hiatt, 2025a)

This repository includes all scripts used for downloading and plotting model inputs, calibration (2011) and validation (2014) analysis, codes to download and produce figures for meteorological, tributary, sea boundary, and diversion inputs for hydrodynamic scenario simulations. The Lagrangian particle tracking model, dorado (Hariharan, 2020), was used to calculate spatial exposure time metrics. dorado was then applied to create maps of the spatial heterogeneity of exposure time for both system-wide and localized metrics. Model descriptions, scenario definitions, model evaluation results, and analytical methods are fully described in the associated publication.



Water exposure time distributions controlled by freshwater releases in a semi-enclosed estuary

Caitlin Turner, Matthew Hiatt

Freshwater diversions manage water shortages, salinity, and control floodwater by redirecting river flows; however, their full ecological and hydrological impact remains unknown. This study examines the Lake Pontchartrain Estuary in Louisiana using a hydrodynamic model and Lagrangian particle tracking to assess how diversion operations (open, closed) and tributary discharge levels (low, median, high) influence water exposure time—the cumulative duration water remains in a domain, including re-entry. Exposure time was analyzed based on the time required for 50\%, 75\%, and 90\% of released particles to leave a defined region of interest (ROI). Results show that when the diversion is open, high tributary discharge reduces exposure times by 51% compared to low discharge. In contrast, when closed, tributary discharge has minimal effect. To identify zones vulnerable to poor water quality due to stagnant water, the spatial heterogeneity of exposure time was evaluated using two metrics: system-wide (time water remains in a system) and localized (time water remains within a ROI) exposure times. The spatial distribution and magnitude of increased exposure times varied between metrics and tributary discharge, highlighting the complexity of transport dynamics. For example, low tributary discharge led to larger isolated zones with longer system-wide and localized exposure times. High tributary discharge created direct flow paths of diversion-sourced water through tidal inlets, short-circuiting the system and creating flow separation. These findings establish a framework for identifying transport mechanisms that influence exposure time and highlighting areas that may be vulnerable to poor water quality.


References:

Hariharan, J., Wright, K., & Passalacqua, P. (2020). dorado: A Python package for simulating passive particle transport in shallow-water flows. Journal of Open Source Software, 5 (54), 2585. doi: 10.21105/joss.02585

Turner, C., M. Hiatt (2025). Water exposure time distributions controlled by freshwater releases in a semi-enclosed estuary published in Water Resources Research WRR 2025 data, HydroShare, http://www.hydroshare.org/10.4211/hs.f1c83ff830bb47c5a7c84e6f5217ea5c
