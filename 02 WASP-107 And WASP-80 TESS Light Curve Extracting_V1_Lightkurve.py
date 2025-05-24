import lightkurve as lk
import os
import matplotlib.pyplot as plt
import inspect
import numpy as np
import astropy.units as u


# Define the directories
root = os.getcwd()
processed_lightcurve_plots_dir = root + "/02 Processed Lightcurve Plots_V1_Lightkurve"
os.makedirs(processed_lightcurve_plots_dir, exist_ok=True)
wasp_107_processed_lightcurve_plots_parent_dir = processed_lightcurve_plots_dir + "/WASP-107"
os.makedirs(wasp_107_processed_lightcurve_plots_parent_dir, exist_ok=True)
wasp_80_processed_lightcurve_plots_parent_dir = processed_lightcurve_plots_dir + "/WASP-80"
os.makedirs(wasp_80_processed_lightcurve_plots_parent_dir, exist_ok=True)


# Define the starting point of TESS observation time
TESS_time = 2457000 # BJD




# WASP-107 SPOC Exptime = 1800s Light Curve Processing
wasp_107_cadence = 1800
wasp_107_processed_lightcurve_plots_dir = wasp_107_processed_lightcurve_plots_parent_dir + f"/Exposure Time={wasp_107_cadence}s"
os.makedirs(wasp_107_processed_lightcurve_plots_dir, exist_ok=True)


# Download and plot the WASP-107 TESS-SPOC raw light curve
i = 1 #count the step
wasp_107_lc_TESS_SPOC_raw = lk.search_lightcurve("WASP-107",author='*SPOC*',exptime= wasp_107_cadence).download()
wasp_107_lc_TESS_SPOC_raw_plot, ax_raw = plt.subplots(figsize=(20, 5))
wasp_107_lc_TESS_SPOC_raw.errorbar(ax=ax_raw)
wasp_107_lc_TESS_SPOC_raw_plot.figure.tight_layout()
wasp_107_lc_TESS_SPOC_raw_plot.figure.savefig(wasp_107_processed_lightcurve_plots_dir + f"/{i:02} WASP-107 TESS-SPOC Raw Lightcurve Exptime={wasp_107_cadence}s.png")


# Flatten the lightcurve and plot the WASP-107 TESS-SPOC flatten light curve
i += 1 #count the step

wasp_107_flatten_window_proportion = 0.05
wasp_107_flatten_window_length = int(wasp_107_lc_TESS_SPOC_raw.time.shape[0] * wasp_107_flatten_window_proportion)

wasp_107_lc_TESS_SPOC_flatten = wasp_107_lc_TESS_SPOC_raw.flatten(window_length=wasp_107_flatten_window_length)
wasp_107_lc_TESS_SPOC_flatten_plot, ax_flatten = plt.subplots(figsize=(20, 5))
wasp_107_lc_TESS_SPOC_flatten.errorbar(ax=ax_flatten)
wasp_107_lc_TESS_SPOC_flatten_plot.figure.tight_layout()
wasp_107_lc_TESS_SPOC_flatten_plot.figure.savefig(wasp_107_processed_lightcurve_plots_dir + f"/{i:02} WASP-107 TESS-SPOC {wasp_107_flatten_window_proportion * 100}% Window Flatten Lightcurve Exptime={wasp_107_cadence}s.png")


# Clip the outliers and plot the WASP-107 TESS-SPOC clipped light curve
i += 1 #count the step

# # Get the default value of the 'sigma' parameter of the 'remove_outliers' method
# method = lk.LightCurve.remove_outliers # Get the method
# signature = inspect.signature(method) # Get the signature of the method
# sigma_default = signature.parameters['sigma'].default # Get the default value of the 'sigma' parameter

wasp_107_sigma = 10

wasp_107_lc_TESS_SPOC_clipped = wasp_107_lc_TESS_SPOC_flatten.remove_outliers(sigma=wasp_107_sigma)
wasp_107_lc_TESS_SPOC_clipped_plot, ax_clipped = plt.subplots(figsize=(20, 5))
wasp_107_lc_TESS_SPOC_clipped.errorbar(ax=ax_clipped)
wasp_107_lc_TESS_SPOC_clipped_plot.figure.tight_layout()
wasp_107_lc_TESS_SPOC_clipped_plot.figure.savefig(wasp_107_processed_lightcurve_plots_dir + f"/{i:02} WASP-107 TESS-SPOC {wasp_107_sigma} Sigma Clipped Lightcurve Exptime={wasp_107_cadence}s.png")


# Fold the lightcurve based on the parameters of WASP-107b and plot the WASP-107 TESS-SPOC folded light curve
i += 1 #count the step

wasp_107b_period = 5.721490 # From NASA Exoplanet Archive

# Retrieve the epoch time
wasp_107b_primary_transit_time = 2457584.329897 - TESS_time # BTJD, from NASA Exoplanet Archive
wasp_107b_epoch_time = wasp_107b_primary_transit_time
n = 0
while True:
    n += 1
    wasp_107b_epoch_time += wasp_107b_period
    if wasp_107b_epoch_time >= wasp_107_lc_TESS_SPOC_clipped.time.value[0]:
        break
wasp_107b_differences = np.abs(wasp_107_lc_TESS_SPOC_clipped.time.value - wasp_107b_epoch_time)
min_diff_index = np.argmin(wasp_107b_differences)
wasp_107b_epoch_time_obs = wasp_107_lc_TESS_SPOC_clipped.time.value[min_diff_index]

wasp_107b_lc_TESS_SPOC_folded = wasp_107_lc_TESS_SPOC_clipped.fold(period=wasp_107b_period, epoch_time=wasp_107b_epoch_time_obs)
wasp_107b_lc_TESS_SPOC_folded_plot, ax_folded = plt.subplots(figsize=(10, 5))
wasp_107b_lc_TESS_SPOC_folded.errorbar(ax=ax_folded)
wasp_107b_lc_TESS_SPOC_folded_plot.figure.tight_layout()
wasp_107b_lc_TESS_SPOC_folded_plot.figure.savefig(wasp_107_processed_lightcurve_plots_dir + f"/{i:02} WASP-107 TESS-SPOC Period={wasp_107b_period}d Folded Lightcurve (For WASP-107b) Exptime={wasp_107_cadence}s.png")


# Bin the lightcurve and plot the WASP-107 TESS-SPOC binned light curve
i += 1 #count the step

wasp_107_frame_bin_size = 2.5
wasp_107_time_bin_size = wasp_107_cadence * u.second * wasp_107_frame_bin_size

wasp_107b_lc_TESS_SPOC_binned = wasp_107b_lc_TESS_SPOC_folded.bin(time_bin_size=wasp_107_time_bin_size)
wasp_107b_lc_TESS_SPOC_binned_plot, ax_binned = plt.subplots(figsize=(10, 5))
wasp_107b_lc_TESS_SPOC_binned.plot(ax=ax_binned)
wasp_107b_lc_TESS_SPOC_binned.errorbar(ax=ax_binned)
wasp_107b_lc_TESS_SPOC_binned_plot.figure.tight_layout()
wasp_107b_lc_TESS_SPOC_binned_plot.figure.savefig(wasp_107_processed_lightcurve_plots_dir + f"/{i:02} WASP-107 TESS-SPOC Frame_Bin_Size={wasp_107_frame_bin_size} Binned Lightcurve (For WASP-107b) Exptime={wasp_107_cadence}s.png")




# WASP-80 SPOC Light Curves Processing
# Define the parameters ------ corresponding to the cadence
wasp_80_cadence_array = [600,120,20]
wasp_80_flatten_window_proportion_array = [0.02,0.05,0.05]
wasp_80_sigma_array = [10,10,10]
wasp_80_frame_bin_size_array = [2.5,10,50]

for j in range(len(wasp_80_cadence_array)):
    wasp_80_cadence = wasp_80_cadence_array[j]
    wasp_80_processed_lightcurve_plots_dir = wasp_80_processed_lightcurve_plots_parent_dir + f"/Exposure Time={wasp_80_cadence}s"
    os.makedirs(wasp_80_processed_lightcurve_plots_dir, exist_ok=True)


    # Download and plot the WASP-80 TESS-SPOC raw light curve
    i = 1 #count the step
    wasp_80_lc_TESS_SPOC_raw = lk.search_lightcurve("WASP-80",author='*SPOC*',exptime= wasp_80_cadence).download()
    wasp_80_lc_TESS_SPOC_raw_plot, ax_raw = plt.subplots(figsize=(20, 5))
    wasp_80_lc_TESS_SPOC_raw.errorbar(ax=ax_raw)
    wasp_80_lc_TESS_SPOC_raw_plot.figure.tight_layout()
    wasp_80_lc_TESS_SPOC_raw_plot.figure.savefig(wasp_80_processed_lightcurve_plots_dir + f"/{i:02} WASP-80 TESS-SPOC Raw Lightcurve Exptime={wasp_80_cadence}s.png")


    # Flatten the lightcurve and plot the WASP-80 TESS-SPOC flatten light curve
    i += 1 #count the step

    wasp_80_flatten_window_proportion = wasp_80_flatten_window_proportion_array[j]
    wasp_80_flatten_window_length = int(wasp_80_lc_TESS_SPOC_raw.time.shape[0] * wasp_80_flatten_window_proportion)

    wasp_80_lc_TESS_SPOC_flatten = wasp_80_lc_TESS_SPOC_raw.flatten(window_length=wasp_80_flatten_window_length)
    wasp_80_lc_TESS_SPOC_flatten_plot, ax_flatten = plt.subplots(figsize=(20, 5))
    wasp_80_lc_TESS_SPOC_flatten.errorbar(ax=ax_flatten)
    wasp_80_lc_TESS_SPOC_flatten_plot.figure.tight_layout()
    wasp_80_lc_TESS_SPOC_flatten_plot.figure.savefig(wasp_80_processed_lightcurve_plots_dir + f"/{i:02} WASP-80 TESS-SPOC {wasp_80_flatten_window_proportion * 100}% Window Flatten Lightcurve Exptime={wasp_80_cadence}s.png")


    # Clip the outliers and plot the WASP-80 TESS-SPOC clipped light curve
    i += 1 #count the step

    # # Get the default value of the 'sigma' parameter of the 'remove_outliers' method
    # method = lk.LightCurve.remove_outliers # Get the method
    # signature = inspect.signature(method) # Get the signature of the method
    # sigma_default = signature.parameters['sigma'].default # Get the default value of the 'sigma' parameter

    wasp_80_sigma = wasp_80_sigma_array[j]

    wasp_80_lc_TESS_SPOC_clipped = wasp_80_lc_TESS_SPOC_flatten.remove_outliers(sigma=wasp_80_sigma)
    wasp_80_lc_TESS_SPOC_clipped_plot, ax_clipped = plt.subplots(figsize=(20, 5))
    wasp_80_lc_TESS_SPOC_clipped.errorbar(ax=ax_clipped)
    wasp_80_lc_TESS_SPOC_clipped_plot.figure.tight_layout()
    wasp_80_lc_TESS_SPOC_clipped_plot.figure.savefig(wasp_80_processed_lightcurve_plots_dir + f"/{i:02} WASP-80 TESS-SPOC {wasp_80_sigma} Sigma Clipped Lightcurve Exptime={wasp_80_cadence}s.png")


    # Fold the lightcurve based on the parameters of WASP-80b and plot the WASP-80 TESS-SPOC folded light curve
    i += 1 #count the step

    wasp_80b_period = 3.067853 # From NASA Exoplanet Archive

    # Retrieve the epoch time
    wasp_80b_primary_transit_time = 2456125.417574 - TESS_time # BTJD, from NASA Exoplanet Archive
    wasp_80b_epoch_time = wasp_80b_primary_transit_time
    n = 0
    while True:
        n += 1
        wasp_80b_epoch_time += wasp_80b_period
        if wasp_80b_epoch_time >= wasp_80_lc_TESS_SPOC_clipped.time.value[0]:
            break
    wasp_80b_differences = np.abs(wasp_80_lc_TESS_SPOC_clipped.time.value - wasp_80b_epoch_time)
    min_diff_index = np.argmin(wasp_80b_differences)
    wasp_80b_epoch_time_obs = wasp_80_lc_TESS_SPOC_clipped.time.value[min_diff_index]

    wasp_80b_lc_TESS_SPOC_folded = wasp_80_lc_TESS_SPOC_clipped.fold(period=wasp_80b_period, epoch_time=wasp_80b_epoch_time_obs)
    wasp_80b_lc_TESS_SPOC_folded_plot, ax_folded = plt.subplots(figsize=(10, 5))
    wasp_80b_lc_TESS_SPOC_folded.errorbar(ax=ax_folded)
    wasp_80b_lc_TESS_SPOC_folded_plot.figure.tight_layout()
    wasp_80b_lc_TESS_SPOC_folded_plot.figure.savefig(wasp_80_processed_lightcurve_plots_dir + f"/{i:02} WASP-80 TESS-SPOC Period={wasp_80b_period}d Folded Lightcurve (For WASP-80b) Exptime={wasp_80_cadence}s.png")


    # Bin the lightcurve and plot the WASP-80 TESS-SPOC binned light curve
    i += 1 #count the step

    wasp_80_frame_bin_size = wasp_80_frame_bin_size_array[j]
    wasp_80_time_bin_size = wasp_80_cadence * u.second * wasp_80_frame_bin_size

    wasp_80b_lc_TESS_SPOC_binned = wasp_80b_lc_TESS_SPOC_folded.bin(time_bin_size=wasp_80_time_bin_size)
    wasp_80b_lc_TESS_SPOC_binned_plot, ax_binned = plt.subplots(figsize=(10, 5))
    wasp_80b_lc_TESS_SPOC_binned.plot(ax=ax_binned)
    wasp_80b_lc_TESS_SPOC_binned.errorbar(ax=ax_binned)
    wasp_80b_lc_TESS_SPOC_binned_plot.figure.tight_layout()
    wasp_80b_lc_TESS_SPOC_binned_plot.figure.savefig(wasp_80_processed_lightcurve_plots_dir + f"/{i:02} WASP-80 TESS-SPOC Frame_Bin_Size={wasp_80_frame_bin_size} Binned Lightcurve (For WASP-80b) Exptime={wasp_80_cadence}s.png")