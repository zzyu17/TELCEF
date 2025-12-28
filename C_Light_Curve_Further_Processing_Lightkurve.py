import os
import warnings
import time

import astropy.units as u
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt

from utils import format_lc_fits_fn_by_provenance, calculate_cdpp, sort_lc
from A_ab_Configuration_Loader import *




# Define the directories
lightkurve_lc_dir = data_dir + config['directory']['lightkurve_lc_dir']
os.makedirs(lightkurve_lc_dir, exist_ok=True)
lightkurve_lc_dir_source = lightkurve_lc_dir + f"/{name}"
os.makedirs(lightkurve_lc_dir_source, exist_ok=True)

lightkurve_processed_lightcurve_plots_dir = base_dir + config["directory"]["lightkurve_processed_lightcurve_plots_dir"]
os.makedirs(lightkurve_processed_lightcurve_plots_dir, exist_ok=True)
lightkurve_processed_lightcurve_plots_dir_source_sector = lightkurve_processed_lightcurve_plots_dir + f"/{name}_Sector-{sector}"
lightkurve_processed_lightcurve_plots_dir_source_sector_suffix = config["directory"]["lightkurve_processed_lightcurve_plots_dir_source_sector_suffix"]
if lightkurve_processed_lightcurve_plots_dir_source_sector_suffix is not None:
    lightkurve_processed_lightcurve_plots_dir_source_sector += f"{lightkurve_processed_lightcurve_plots_dir_source_sector_suffix}"
os.makedirs(lightkurve_processed_lightcurve_plots_dir_source_sector, exist_ok=True)
lightkurve_processed_lightcurve_plots_suffix = config["directory"]["lightkurve_processed_lightcurve_plots_suffix"] if config["directory"]["lightkurve_processed_lightcurve_plots_suffix"] is not None else ""




### ------ Lightcurve Selecting ------ ###
# Set the lightcurve selecting and reading parameters
lc_raw_provenance = config['lightkurve']['lightcurve_provenance']
flux_column = config['lightkurve']['flux_column']
if lc_raw_provenance == "lightkurve":
    raise ValueError("The 'lightcurve_provenance' parameter can't be set to 'lightkurve' in C_Light_Curve_Further_Processing_Lightkurve. Please set it to 'extracted', 'downloaded' or 'eleanor' in the configuration file.")

lc_raw_fn = format_lc_fits_fn_by_provenance(lc_raw_provenance, config)

# Search for the lightcurve file in the data directory
fits_path_list = [os.path.join(root, f) for root, _, files in os.walk(data_dir) for f in files if f.endswith('.fits')]
fits_fn_list = [os.path.basename(fits_path) for fits_path in fits_path_list]
lc_raw_path = None
for l in range(len(fits_fn_list)):
    if fits_fn_list[l] == lc_raw_fn:
        lc_raw_path = fits_path_list[l]


# Read the lightcurve file via Lightkurve
if lc_raw_path is None:
    raise FileNotFoundError("The specified lightcurve file does not exist in the data directory. Please check if the file exists and run the light curve extracting/downloading scripts first if necessary.")
else:
    lc_raw = lk.read(lc_raw_path, flux_column=flux_column)
    print(f"Successfully found and read the specified lightcurve file: {lc_raw_path}.\n")




# Define the lightcurve-specified directory
lc_raw_fn_pure = os.path.splitext(lc_raw_fn)[0]
lc_raw_fn_suffix = lc_raw_fn_pure.replace(f"{name}_{mission}_Sector-{sector}_", "", 1).replace(f"_LC", "", -1)

lightkurve_processed_lightcurve_plots_dir_source_sector_lc = lightkurve_processed_lightcurve_plots_dir_source_sector + f"/{lc_raw_fn_suffix}"
os.makedirs(lightkurve_processed_lightcurve_plots_dir_source_sector_lc, exist_ok=True)


# Define the lightcurve plot title
lc_plot_title = lc_raw_fn_pure.replace(f"_{mission}", "", 1).replace("_LC", "", -1).replace("_", " ").replace("Sector-", "Sector ").replace("lightkurve aperture", "lightkurve_aperture")


# Set whether to plot the error bar of the light curve
plot_errorbar = config["lightkurve"]["plot_errorbar"]




### ------ Remove NaNs ------ ###
i = 1 # count the step

# Remove NaNs and plot the light curve
j = 1 # count the sub-step
lc_raw_nans_removed = sort_lc(lc_raw.remove_nans()) # remove NaNs from the raw light curve and sort it in strictly increasing time order
lc_raw_nans_removed_cdpp = calculate_cdpp(lc_raw_nans_removed, exptime=exptime, cdpp_transit_duration=cdpp_transit_duration)

lc_raw_nans_removed_plot, ax_lc_raw_nans_removed = plt.subplots(figsize=(20, 5))
if plot_errorbar:
    lc_raw_nans_removed.scatter(ax=ax_lc_raw_nans_removed, label=None, s=0.1)
    lc_raw_nans_removed.errorbar(ax=ax_lc_raw_nans_removed, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_raw_nans_removed_cdpp:.2f} ppm")
else:
    lc_raw_nans_removed.scatter(ax=ax_lc_raw_nans_removed, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_raw_nans_removed_cdpp:.2f} ppm", s=0.1)
ax_lc_raw_nans_removed.set_title(f"{lc_plot_title} NaNs-Removed Raw Light Curve")
lc_raw_nans_removed_plot.figure.tight_layout()
lc_raw_nans_removed_plot.figure.savefig(lightkurve_processed_lightcurve_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_NaNs-Removed_Raw_Light_Curve{lightkurve_processed_lightcurve_plots_suffix}.png")
plt.close()


# Fit the NaNs-removed raw light curve using the Box Least Squares (BLS) method
j += 1 # count the sub-step
pg_bls_raw_nans_removed_start_time = time.time() # measure the start time
p_bls_raw_nans_removed_range = np.linspace(config['lightkurve']['p_bls_raw_nans_removed_range'][0], config['lightkurve']['p_bls_raw_nans_removed_range'][1], 10000) ##### set the range of period to search #####
pg_bls_raw_nans_removed = lc_raw_nans_removed.to_periodogram(method='bls', period=p_bls_raw_nans_removed_range, frequency_factor=500)
pg_bls_raw_nans_removed_end_time = time.time() # measure the end time
pg_bls_raw_nans_removed_fitting_time = pg_bls_raw_nans_removed_end_time - pg_bls_raw_nans_removed_start_time # calculate the fitting time
print(f"Fitted global parameters of {lc_plot_title} NaNs-Removed Raw light curve using the Box Least Squares (BLS) method in {pg_bls_raw_nans_removed_fitting_time} seconds.\n")

pg_bls_raw_nans_removed_plot, ax_pg_bls_raw_nans_removed = plt.subplots(figsize=(20, 5))
pg_bls_raw_nans_removed.plot(ax=ax_pg_bls_raw_nans_removed)
ax_pg_bls_raw_nans_removed.set_title(f"{lc_plot_title} NaNs-Removed Raw BLS Periodogram")
pg_bls_raw_nans_removed_plot.figure.tight_layout()
pg_bls_raw_nans_removed_plot.figure.savefig(lightkurve_processed_lightcurve_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_NaNs-Removed_Raw_BLS_Periodogram{lightkurve_processed_lightcurve_plots_suffix}.png")
plt.close()


j += 1 # count the sub-step
p_bls_raw_nans_removed = float(pg_bls_raw_nans_removed.period_at_max_power.value)
t0_bls_raw_nans_removed = float(pg_bls_raw_nans_removed.transit_time_at_max_power.value)
transit_duration_bls_raw_nans_removed = float(pg_bls_raw_nans_removed.duration_at_max_power.value)
transit_depth_bls_raw_nans_removed = float(pg_bls_raw_nans_removed.depth_at_max_power.value / np.nanmedian(lc_raw_nans_removed.flux.value)) # calculate the normalized transit depth
k_bls_raw_nans_removed = float((transit_depth_bls_raw_nans_removed)**0.5)
transit_mask_bls_raw_nans_removed_span_coefficient = config['lightkurve']['transit_mask_bls_raw_nans_removed_span_coefficient'] if config['lightkurve']['transit_mask_bls_raw_nans_removed_span_coefficient'] is not None else 1.8 ##### set the coefficient of BLS transit mask span #####
transit_mask_bls_raw_nans_removed = pg_bls_raw_nans_removed.get_transit_mask(period=p_bls_raw_nans_removed, transit_time=t0_bls_raw_nans_removed, duration=transit_duration_bls_raw_nans_removed * transit_mask_bls_raw_nans_removed_span_coefficient)
lc_fitted_bls_raw_nans_removed = pg_bls_raw_nans_removed.get_transit_model(period=p_bls_raw_nans_removed, transit_time=t0_bls_raw_nans_removed, duration=transit_duration_bls_raw_nans_removed)

lc_fitted_bls_raw_nans_removed_plot, ax_lc_fitted_bls_raw_nans_removed = plt.subplots(figsize=(20, 5))
if plot_errorbar:
    lc_raw_nans_removed.scatter(ax=ax_lc_fitted_bls_raw_nans_removed, label=None, s=0.1)
    lc_raw_nans_removed.errorbar(ax=ax_lc_fitted_bls_raw_nans_removed, label=f"NaNs-Removed Raw Light Curve")
else:
    lc_raw_nans_removed.scatter(ax=ax_lc_fitted_bls_raw_nans_removed, label=f"NaNs-Removed Raw Light Curve", s=0.1)
lc_fitted_bls_raw_nans_removed.plot(ax=ax_lc_fitted_bls_raw_nans_removed, c='red', label=f"Best Fitted BLS Model")
ax_lc_fitted_bls_raw_nans_removed.set_title(f"{lc_plot_title} BLS Best Fitted NaNs-Removed Raw Light Curve")
lc_fitted_bls_raw_nans_removed_plot.figure.tight_layout()
lc_fitted_bls_raw_nans_removed_plot.figure.savefig(lightkurve_processed_lightcurve_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_BLS_Best_Fitted_NaNs-Removed_Raw_Light_Curve{lightkurve_processed_lightcurve_plots_suffix}.png")
plt.close()


# Retrieve and plot Nans-removed raw baseline
j += 1 # count the sub-step
lc_raw_nans_removed_baseline = lc_raw_nans_removed[~transit_mask_bls_raw_nans_removed] # retrieve the out-of-transit light curve (baseline)
lc_raw_nans_removed_transit = lc_raw_nans_removed[transit_mask_bls_raw_nans_removed] # retrieve the in-transit light curve
lc_raw_nans_removed_baseline_cdpp = calculate_cdpp(lc_raw_nans_removed_baseline, exptime=exptime, cdpp_transit_duration=cdpp_transit_duration)

lc_raw_nans_removed_baseline_plot, ax_lc_raw_nans_removed_baseline = plt.subplots(figsize=(20, 5))
if plot_errorbar:
    lc_raw_nans_removed_baseline.scatter(ax=ax_lc_raw_nans_removed_baseline, label=None, s=0.1)
    lc_raw_nans_removed_baseline.errorbar(ax=ax_lc_raw_nans_removed_baseline, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_raw_nans_removed_baseline_cdpp:.2f} ppm")
else:
    lc_raw_nans_removed_baseline.scatter(ax=ax_lc_raw_nans_removed_baseline, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_raw_nans_removed_baseline_cdpp:.2f} ppm", s=0.1)
ax_lc_raw_nans_removed_baseline.set_title(f"{lc_plot_title} NaNs-Removed Raw Baseline")
lc_raw_nans_removed_baseline_plot.figure.tight_layout()
lc_raw_nans_removed_baseline_plot.figure.savefig(lightkurve_processed_lightcurve_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_NaNs-Removed_Raw_Baseline{lightkurve_processed_lightcurve_plots_suffix}.png")
plt.close()

config = update_config(config_path, {'lightkurve.lc_raw_nans_removed_cdpp': lc_raw_nans_removed_cdpp,
                                     'lightkurve.lc_raw_nans_removed_baseline_cdpp': lc_raw_nans_removed_baseline_cdpp,
                                     'lightkurve.raw_nans_removed_bls_fitted_parameters.k': k_bls_raw_nans_removed,
                                     'lightkurve.raw_nans_removed_bls_fitted_parameters.p': p_bls_raw_nans_removed,
                                     'lightkurve.raw_nans_removed_bls_fitted_parameters.t0': t0_bls_raw_nans_removed,
                                     'lightkurve.raw_nans_removed_bls_fitted_parameters.transit_duration': transit_duration_bls_raw_nans_removed,
                                     'lightkurve.raw_nans_removed_bls_fitted_parameters.transit_depth': transit_depth_bls_raw_nans_removed})




### ------ Flatten ------ ###
i += 1 # count the step


# Set the flattening parameters
flatten = config['lightkurve']['flatten']
flatten_window_proportion = config['lightkurve']['flatten_window_proportion']
flatten_polyorder = config['lightkurve']['flatten_polyorder']

flatten_window_length = int(flatten_window_proportion * lc_raw_nans_removed.time.shape[0])
if flatten_window_length % 2 == 0:
    flatten_window_length += 1  # the flatten window length should be an odd number


if flatten:
    # Flatten and plot the light curve
    lc_flattened, lc_flattened_trend = lc_raw_nans_removed.flatten(window_length=flatten_window_length, polyorder=flatten_polyorder, mask=transit_mask_bls_raw_nans_removed, return_trend=True)
    lc_flattened_cdpp = calculate_cdpp(lc_flattened, exptime=exptime, cdpp_transit_duration=cdpp_transit_duration)

    j = 1 # count the sub-step
    lc_flattened_plot, ax_lc_flattened = plt.subplots(figsize=(20, 5))
    if plot_errorbar:
        lc_flattened.scatter(ax=ax_lc_flattened, label=None, s=0.1)
        lc_flattened.errorbar(ax=ax_lc_flattened, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_flattened_cdpp:.2f} ppm")
    else:
        lc_flattened.scatter(ax=ax_lc_flattened, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_flattened_cdpp:.2f} ppm", s=0.1)
    ax_lc_flattened.set_title(f"{lc_plot_title} {flatten_window_proportion * 100:.1f}% Window Flattened Light Curve")
    ax_lc_flattened.set_ylabel("Flux")
    lc_flattened_plot.figure.tight_layout()
    lc_flattened_plot.figure.savefig(lightkurve_processed_lightcurve_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_{flatten_window_proportion * 100:.1f}%_Window_Flattened_Light_Curve{lightkurve_processed_lightcurve_plots_suffix}.png")
    plt.close()

    j += 1 # count the sub-step
    lc_flattened_trend_plot, (ax_lc_flattened, ax_lc_flattened_trend) = plt.subplots(2, 1, figsize=(20, 10))
    if plot_errorbar:
        lc_flattened.scatter(ax=ax_lc_flattened, label=None, s=0.1)
        lc_flattened.errorbar(ax=ax_lc_flattened, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_flattened_cdpp:.2f} ppm")
    else:
        lc_flattened.scatter(ax=ax_lc_flattened, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_flattened_cdpp:.2f} ppm", s=0.1)
    ax_lc_flattened.set_title(f"{lc_plot_title} {flatten_window_proportion * 100:.1f}% Window Flattened Light Curve")
    ax_lc_flattened.set_ylabel("Flux")
    lc_flattened_trend.plot(ax=ax_lc_flattened_trend)
    ax_lc_flattened_trend.set_title(f"{lc_plot_title} {flatten_window_proportion * 100:.1f}% Window Flatten Trend")
    lc_flattened_trend_plot.figure.tight_layout()
    lc_flattened_trend_plot.figure.savefig(lightkurve_processed_lightcurve_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_{flatten_window_proportion * 100:.1f}%_Window_Flatten_Trend{lightkurve_processed_lightcurve_plots_suffix}.png")
    plt.close()


    # Retrieve and plot the flattened baseline
    lc_flattened_baseline = lc_flattened[~transit_mask_bls_raw_nans_removed]
    lc_flattened_transit = lc_flattened[transit_mask_bls_raw_nans_removed]
    lc_flattened_baseline_trend = lc_flattened_trend[~transit_mask_bls_raw_nans_removed]
    lc_flattened_transit_trend = lc_flattened_trend[transit_mask_bls_raw_nans_removed]
    lc_flattened_baseline_cdpp = calculate_cdpp(lc_flattened_baseline, exptime=exptime, cdpp_transit_duration=cdpp_transit_duration)

    j += 1 # count the sub-step
    lc_flattened_baseline_plot, ax_lc_flattened_baseline = plt.subplots(figsize=(20, 5))
    if plot_errorbar:
        lc_flattened_baseline.scatter(ax=ax_lc_flattened_baseline, label=None, s=0.1)
        lc_flattened_baseline.errorbar(ax=ax_lc_flattened_baseline, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_flattened_baseline_cdpp:.2f} ppm")
    else:
        lc_flattened_baseline.scatter(ax=ax_lc_flattened_baseline, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_flattened_baseline_cdpp:.2f} ppm", s=0.1)
    ax_lc_flattened_baseline.set_title(f"{lc_plot_title} {flatten_window_proportion * 100:.1f}% Window Flattened Baseline Light Curve")
    ax_lc_flattened_baseline.set_ylabel("Flux")
    lc_flattened_baseline_plot.figure.tight_layout()
    lc_flattened_baseline_plot.figure.savefig(lightkurve_processed_lightcurve_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_{flatten_window_proportion * 100:.1f}%_Window_Flattened_Baseline_Light_Curve{lightkurve_processed_lightcurve_plots_suffix}.png")
    plt.close()

    j += 1 # count the sub-step
    lc_flattened_baseline_trend_plot, (ax_lc_flattened_baseline, ax_lc_flattened_baseline_trend) = plt.subplots(2, 1, figsize=(20, 10))
    if plot_errorbar:
        lc_flattened_baseline.scatter(ax=ax_lc_flattened_baseline, label=None, s=0.1)
        lc_flattened_baseline.errorbar(ax=ax_lc_flattened_baseline, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_flattened_baseline_cdpp:.2f} ppm")
    else:
        lc_flattened_baseline.scatter(ax=ax_lc_flattened_baseline, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_flattened_baseline_cdpp:.2f} ppm", s=0.1)
    ax_lc_flattened_baseline.set_title(f"{lc_plot_title} {flatten_window_proportion * 100:.1f}% Window Flattened Baseline Light Curve")
    ax_lc_flattened_baseline.set_ylabel("Flux")
    lc_flattened_baseline_trend.plot(ax=ax_lc_flattened_baseline_trend)
    ax_lc_flattened_baseline_trend.set_title(f"{lc_plot_title} {flatten_window_proportion * 100:.1f}% Window Flattened Baseline Trend")
    lc_flattened_baseline_trend_plot.figure.tight_layout()
    lc_flattened_baseline_trend_plot.figure.savefig(lightkurve_processed_lightcurve_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_{flatten_window_proportion * 100:.1f}%_Window_Flattened_Baseline_Trend{lightkurve_processed_lightcurve_plots_suffix}.png")
    plt.close()


else:
    lc_flattened_baseline = lc_raw_nans_removed_baseline.copy()
    lc_flattened_baseline_cdpp = calculate_cdpp(lc_flattened_baseline, exptime=exptime, cdpp_transit_duration=cdpp_transit_duration)

    lc_flattened_transit = lc_raw_nans_removed_transit.copy()

    lc_flattened = lc_raw_nans_removed.copy()
    lc_flattened_cdpp = calculate_cdpp(lc_flattened, exptime=exptime, cdpp_transit_duration=cdpp_transit_duration)

config = update_config(config_path, {'lightkurve.flatten_window_length': flatten_window_length,
                                     'lightkurve.lc_flattened_cdpp': lc_flattened_cdpp,
                                     'lightkurve.lc_flattened_baseline_cdpp': lc_flattened_baseline_cdpp})




### ------ Clip ------ ###
i += 1 # count the step


# Set the clipping parameters
clip = config['lightkurve']['clip']
clip_transit = config['lightkurve']['clip_transit']
sigma_baseline = config['lightkurve']['sigma_baseline']
sigma_transit_upper = config['lightkurve']['sigma_transit_upper']
sigma_transit_lower = config['lightkurve']['sigma_transit_lower'] if config['lightkurve']['sigma_transit_lower'] is not None else float('inf')


if clip:
    # Clip and plot the baseline
    j = 1 # count the sub-step
    lc_clipped_baseline = lc_flattened_baseline.remove_outliers(sigma=sigma_baseline)
    lc_clipped_baseline_cdpp = calculate_cdpp(lc_clipped_baseline, exptime=exptime, cdpp_transit_duration=cdpp_transit_duration)

    lc_clipped_baseline_plot, ax_lc_clipped_baseline = plt.subplots(figsize=(20, 5))
    if plot_errorbar:
        lc_clipped_baseline.scatter(ax=ax_lc_clipped_baseline, label=None, s=0.1)
        lc_clipped_baseline.errorbar(ax=ax_lc_clipped_baseline, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_clipped_baseline_cdpp:.2f} ppm")
    else:
        lc_clipped_baseline.scatter(ax=ax_lc_clipped_baseline, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_clipped_baseline_cdpp:.2f} ppm", s=0.1)
    ax_lc_clipped_baseline.set_title(f"{lc_plot_title} {sigma_baseline:.1f} Sigma Clipped Baseline")
    ax_lc_clipped_baseline.set_ylabel("Flux")
    lc_clipped_baseline_plot.figure.tight_layout()
    lc_clipped_baseline_plot.figure.savefig(lightkurve_processed_lightcurve_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_{sigma_baseline:.1f}_Sigma_Clipped_Baseline{lightkurve_processed_lightcurve_plots_suffix}.png")
    plt.close()


    # Clip and plot the transit if there are upper outliers
    if clip_transit:
        j += 1 # count the sub-step
        lc_flattened_baseline_nanmedian = np.nanmedian(lc_flattened_baseline.flux)
        lc_clipped_transit = lc_flattened_transit.remove_outliers(cenfunc=lambda x, **kwargs: lc_flattened_baseline_nanmedian, sigma_lower=sigma_transit_lower, sigma_upper=sigma_transit_upper) # remove the upper outliers in the transit based on the nanmedian of lc_flattened_baseline

        lc_clipped_transit_plot, ax_lc_clipped_transit = plt.subplots(figsize=(20, 5))
        if plot_errorbar:
            lc_clipped_transit.scatter(ax=ax_lc_clipped_transit, s=0.1)
            lc_clipped_transit.errorbar(ax=ax_lc_clipped_transit)
        else:
            lc_clipped_transit.scatter(ax=ax_lc_clipped_transit, s=0.1)
        ax_lc_clipped_transit.set_title(f"{lc_plot_title} {sigma_transit_upper:.1f} Upper Sigma Clipped Transit")
        ax_lc_clipped_transit.set_ylabel("Flux")
        lc_clipped_transit_plot.figure.tight_layout()
        lc_clipped_transit_plot.figure.savefig(lightkurve_processed_lightcurve_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_{sigma_transit_upper:.1f}_Upper_Sigma_Clipped_Transit{lightkurve_processed_lightcurve_plots_suffix}.png")
        plt.close()

    else:
        lc_clipped_transit = lc_flattened_transit.copy()


    # Retrieve and plot the clipped light curve
    j += 1 # count the sub-step
    lc_clipped = sort_lc(lc_clipped_baseline.append(lc_clipped_transit)) # append the in-transit light curve to the baseline to retrieve the clipped light curve
    lc_clipped_cdpp = calculate_cdpp(lc_clipped, exptime=exptime, cdpp_transit_duration=cdpp_transit_duration)

    lc_clipped_plot, ax_lc_clipped = plt.subplots(figsize=(20, 5))
    if plot_errorbar:
        lc_clipped.scatter(ax=ax_lc_clipped, label=None, s=0.1)
        lc_clipped.errorbar(ax=ax_lc_clipped, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_clipped_cdpp:.2f} ppm")
    else:
        lc_clipped.scatter(ax=ax_lc_clipped, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_clipped_cdpp:.2f} ppm", s=0.1)
    ax_lc_clipped.set_title(f"{lc_plot_title} {sigma_baseline:.1f} Sigma Clipped Light Curve")
    ax_lc_clipped.set_ylabel("Flux")
    lc_clipped_plot.figure.tight_layout()
    lc_clipped_plot.figure.savefig(lightkurve_processed_lightcurve_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_{sigma_baseline:.1f}_Sigma_Clipped_Light_Curve{lightkurve_processed_lightcurve_plots_suffix}.png")
    plt.close()


else:
    lc_clipped_baseline = lc_flattened_baseline.copy()
    lc_clipped_baseline_cdpp = calculate_cdpp(lc_clipped_baseline, exptime=exptime, cdpp_transit_duration=cdpp_transit_duration)

    lc_clipped_transit = lc_flattened_transit.copy()

    lc_clipped = lc_flattened.copy()
    lc_clipped_cdpp = calculate_cdpp(lc_clipped, exptime=exptime, cdpp_transit_duration=cdpp_transit_duration)

config = update_config(config_path, {'lightkurve.lc_clipped_cdpp': lc_clipped_cdpp,
                                     'lightkurve.lc_clipped_baseline_cdpp': lc_clipped_baseline_cdpp})




# Save the lightkurve processed lightcurve into a FITS file
lightkurve_lc = lc_clipped.copy()
lightkurve_lc_baseline = lc_clipped_baseline.copy()
lightkurve_lc_transit = lc_clipped_transit.copy()

sigma_transit_lower_correction = f'{sigma_transit_lower:.1f}' if sigma_transit_lower != float('inf') else 'inf'


if flatten and clip:
    correction = f"_Flattened-p{flatten_window_proportion:.2f}-l{flatten_window_length}-o{flatten_polyorder}_&_Clipped-b{sigma_baseline:.1f}-tu{sigma_transit_upper:.1f}-tl{sigma_transit_lower_correction}"
elif flatten and not clip:
    correction = f"_Flattened-p{flatten_window_proportion:.2f}-l{flatten_window_length}-o{flatten_polyorder}"
elif not flatten and clip:
    correction = f"_Clipped-b{sigma_baseline:.1f}-tu{sigma_transit_upper:.1f}-tl{sigma_transit_lower_correction}"
else:
    correction = ""

if lc_raw_provenance == "downloaded" and flux_column is not None and flux_column.lower() != "pdcsap_flux":
    correction += f"_{flux_column}".strip("_flux").capitalize()

config = update_config(config_path, {'lightkurve.correction': correction})

lightkurve_lc_fn = lc_raw_fn.replace("_LC", f"{correction}_LC", -1)
lightkurve_lc_path = lightkurve_lc_dir_source + f"/{lightkurve_lc_fn}"
lightkurve_lc.to_fits(path=lightkurve_lc_path, overwrite=True)




print(f"Successfully processed the light curve via Lightkurve and saved it to the data directory of the source: {lightkurve_lc_path}.\n")