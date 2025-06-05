import os
import time

import astropy.units as u
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np




### ------ Preparations ------ ###
# Set the single sector and pipeline to be processed
sector = 54 ##### set the single sector to be processed #####
pipeline = 'SPOC' ##### set the pipeline to be processed #####


# Define the starting point of TESS observation time
TESS_time = 2457000 # BJD

##### Define the source parameters #####
name = "WASP-80"
tic = 243921117
coords = (303.16737208333325,-2.144218611111111)
gaia = 4223507222112425344
tess_mag = 10.3622
# From NASA Exoplanet Archive
period_NASA = 3.06785234 # days
epoch_time_NASA = 2456487.425006 # BJD
transit_duration_NASA = (2.1310 * u.hour).to(u.day).value  # hours to days
transit_depth_NASA = (0.17137) ** 2

# ##### Define the source parameters #####
# name = "WASP-107"
# tic = 429302040
# coords = (188.38685041666665, -10.146173611111111)
# gaia = 3578638842054261248
# tess_mag = 10.418

##### Define the Lightkurve parameters depending on the exptime #####
exptime = 20
exptime_in_days = (exptime * u.second).to(u.day).value  # convert exposure time to days

# set the coefficient of transit mask span manually
transit_mask_raw_nans_removed_coefficient = 1.8

flatten = True
flatten_window_proportion = 0.05
polyorder = 3

clip = True
clip_transit = True
sigma_baseline = 3.0
sigma_upper_transit = 0.4 # set sigma_upper_transit manually
sigma_lower_transit = np.inf # set to np.inf to avoid clipping the real transit

bin = True
cadence_bin_size = 2.5




# Define the calculate_cdpp() function
def calculate_cdpp(lc, transit_duration=13):
    """
    Calculate the Combined Differential Photometric Precision (CDPP) noise metric
    directly from the raw light curve, without applying flattening or outlier removal.

    This method computes the CDPP by:
    1. Normalizing the flux to parts-per-million (ppm) units
    2. Applying a running mean with a window size equal to the transit duration
    3. Calculating the standard deviation of the running mean series

    Parameters
    ----------
    lc : lightkurve.LightCurve
        The light curve object. Must contain 'flux' and 'flux_err' columns.
    transit_duration : int, optional
        The transit duration in units of number of cadences. Default is 13
        (equivalent to 6.5 hours for Kepler 30-min cadence).

    Returns
    -------
    cdpp : float
        The CDPP noise metric in parts-per-million (ppm).

    Notes
    -----
    This method differs from LightCurve.estimate_cdpp() in that it:
    - Does NOT apply Savitzky-Golay filtering (flattening)
    - Does NOT perform sigma-clipping to remove outliers
    - Works directly on the input light curve flux values
    """
    from lightkurve.utils import running_mean

    if not isinstance(transit_duration, int):
        raise ValueError("transit_duration must be an integer in units number of cadences, got {}.".format(transit_duration))

    cleaned_lc = lc.remove_nans()

    normalized_lc = cleaned_lc.normalize("ppm")

    mean = running_mean(data=normalized_lc.flux, window_size=transit_duration)
    return np.std(mean)




# Define the method name based on the script name
script_path = os.path.abspath(__file__)
script_name = str(os.path.basename(script_path))
last_underscore_idx = script_name.rfind('_')
dot_idx = script_name.rfind('.')
if last_underscore_idx != -1 and dot_idx != -1 and last_underscore_idx < dot_idx:
    method = script_name[last_underscore_idx + 1: dot_idx]
else:
    print("Unresolvable script name. Please define the method name manually.")
    method = "Lightkurve" # define the method name manually if the script name is unresolvable

# Define the directories
root = os.getcwd()
processed_lightcurve_plots_dir = root + f"/02 Processed Light Curve Plots_V1_{method}"
os.makedirs(processed_lightcurve_plots_dir, exist_ok=True)
processed_lightcurve_plots_parent_dir = processed_lightcurve_plots_dir + f"/{name}_Sector {sector}_{pipeline}"
os.makedirs(processed_lightcurve_plots_parent_dir, exist_ok=True)
processed_lightcurve_plots_exptime_parent_dir = processed_lightcurve_plots_parent_dir + f"/Exposure Time={exptime}s"
os.makedirs(processed_lightcurve_plots_exptime_parent_dir, exist_ok=True)

eleanor_root = os.path.expanduser("~/.eleanor")
eleanor_root_targetdata = eleanor_root + f"/targetdata"
os.makedirs(eleanor_root_targetdata, exist_ok=True)
eleanor_root_targetdata_source = eleanor_root_targetdata + f"/{name}"
os.makedirs(eleanor_root_targetdata_source, exist_ok=True)




### ------ Lightkurve ------ ###
### Raw ###
i = 1 # count the step


# Download and plot the raw light curve
j = 1 # count the sub-step
lc_raw = lk.search_lightcurve(name, sector=sector, author=f'*{pipeline}*', exptime=exptime).download()
print(f"Downloaded and processing {name} Sector {sector} {pipeline} Exptime={exptime}s Light Curve...\n")
lc_raw_cdpp = calculate_cdpp(lc_raw, transit_duration=216) ##### set the transit duration after fitting transit parameters for the first time #####

lc_raw_plot, ax_lc_raw = plt.subplots(figsize=(20, 5))
lc_raw.errorbar(ax=ax_lc_raw, label=f"simplified CDPP={lc_raw_cdpp:.2f}")
ax_lc_raw.set_title(f"{name} Sector {sector} {pipeline} Raw Light Curve Exptime={exptime}s")
lc_raw_plot.figure.tight_layout()
lc_raw_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {pipeline} Raw Light Curve Exptime={exptime}s.png")


j += 1 # count the sub-step
lc_raw_nans_removed = lc_raw.remove_nans() # Remove NaNs from the raw light curve
lc_raw_nans_removed_cdpp = calculate_cdpp(lc_raw_nans_removed, transit_duration=216) ##### set the transit duration after fitting transit parameters for the first time #####

lc_raw_nans_removed_plot, ax_lc_raw_nans_removed = plt.subplots(figsize=(20, 5))
lc_raw_nans_removed.errorbar(ax=ax_lc_raw_nans_removed, label=f"simplified CDPP={lc_raw_nans_removed_cdpp:.2f}")
ax_lc_raw_nans_removed.set_title(f"{name} Sector {sector} {pipeline} NaNs-Removed Raw Light Curve Exptime={exptime}s")
lc_raw_nans_removed_plot.figure.tight_layout()
lc_raw_nans_removed_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {pipeline} NaNs-Removed Raw Light Curve Exptime={exptime}s.png")


# Fit the NaNs-removed raw light curve using the Box Least Squares (BLS) method
j += 1 # count the sub-step
pg_bls_raw_nans_removed_start_time = time.time() # measure the start time
period_raw_nans_removed_range = np.linspace(0.5, 27, 10000) ##### set the range of period to search #####
pg_bls_raw_nans_removed = lc_raw_nans_removed.to_periodogram(method='bls', period=period_raw_nans_removed_range, frequency_factor=500)
pg_bls_raw_nans_removed_end_time = time.time() # measure the end time
pg_bls_raw_nans_removed_fitting_time = pg_bls_raw_nans_removed_end_time - pg_bls_raw_nans_removed_start_time # calculate the fitting time
print(f"Fitted {name} Sector {sector} {pipeline} NaNs-Removed Raw Light Curve Exptime={exptime}s using the Box Least Squares (BLS) method in {pg_bls_raw_nans_removed_fitting_time} seconds.\n")

pg_bls_raw_nans_removed_plot, ax_pg_bls_raw_nans_removed = plt.subplots(figsize=(20, 5))
pg_bls_raw_nans_removed.plot(ax=ax_pg_bls_raw_nans_removed)
ax_pg_bls_raw_nans_removed.set_title(f"{name} Sector {sector} {pipeline} NaNs-Removed Raw BLS Periodogram Exptime={exptime}s")
pg_bls_raw_nans_removed_plot.figure.tight_layout()
pg_bls_raw_nans_removed_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {pipeline} NaNs-Removed Raw BLS Periodogram Exptime={exptime}s.png")


j += 1 # count the sub-step
period_raw_nans_removed = pg_bls_raw_nans_removed.period_at_max_power.value
epoch_time_raw_nans_removed = pg_bls_raw_nans_removed.transit_time_at_max_power.value
transit_duration_raw_nans_removed = pg_bls_raw_nans_removed.duration_at_max_power.value
transit_duration_raw_nans_removed_proportion = transit_duration_raw_nans_removed / (lc_raw_nans_removed.time.value[-1] - lc_raw_nans_removed.time.value[0])
transit_duration_raw_nans_removed_in_cadence = int(round(transit_duration_raw_nans_removed / exptime_in_days))  # convert transit duration to number of cadences
transit_depth_raw_nans_removed = pg_bls_raw_nans_removed.depth_at_max_power.value / np.nanmedian(lc_raw_nans_removed.flux.value) # calculate the normalized transit depth
transit_mask_raw_nans_removed = pg_bls_raw_nans_removed.get_transit_mask(period=period_raw_nans_removed, transit_time=epoch_time_raw_nans_removed, duration=transit_duration_raw_nans_removed * transit_mask_raw_nans_removed_coefficient)
lc_bls_transit_model_raw_nans_removed = pg_bls_raw_nans_removed.get_transit_model(period=period_raw_nans_removed, transit_time=epoch_time_raw_nans_removed, duration=transit_duration_raw_nans_removed)

lc_bls_transit_model_raw_nans_removed_plot, ax_lc_bls_transit_model_raw_nans_removed = plt.subplots(figsize=(20, 5))
lc_raw_nans_removed.errorbar(ax=ax_lc_bls_transit_model_raw_nans_removed, label=f"NaNs-Removed Raw Light Curve Exptime={exptime}s")
lc_bls_transit_model_raw_nans_removed.plot(ax=ax_lc_bls_transit_model_raw_nans_removed, c='red', label=f"NaNs-Removed Raw BLS Transit Model Exptime={exptime}s")
ax_lc_bls_transit_model_raw_nans_removed.set_title(f"{name} Sector {sector} {pipeline} NaNs-Removed Raw Light Curve & BLS Transit Model Exptime={exptime}s")
lc_bls_transit_model_raw_nans_removed_plot.figure.tight_layout()
lc_bls_transit_model_raw_nans_removed_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {pipeline} NaNs-Removed Raw Light Curve & BLS Transit Model Exptime={exptime}s.png")


# Retrieve and plot Nans-removed raw baseline
j += 1 # count the sub-step
lc_raw_nans_removed_baseline = lc_raw_nans_removed[~transit_mask_raw_nans_removed] # retrieve the out-of-transit light curve (baseline)
lc_raw_nans_removed_transit = lc_raw_nans_removed[transit_mask_raw_nans_removed] # retrieve the in-transit light curve
lc_raw_nans_removed_baseline_cdpp = calculate_cdpp(lc_raw_nans_removed_baseline, transit_duration=transit_duration_raw_nans_removed_in_cadence)

lc_raw_nans_removed_baseline_plot, ax_lc_raw_nans_removed_baseline = plt.subplots(figsize=(20, 5))
lc_raw_nans_removed_baseline.errorbar(ax=ax_lc_raw_nans_removed_baseline, label=f"simplified CDPP={lc_raw_nans_removed_baseline_cdpp:.2f}")
ax_lc_raw_nans_removed_baseline.set_title(f"{name} Sector {sector} {pipeline} NaNs-Removed Raw Baseline Exptime={exptime}s")
lc_raw_nans_removed_baseline_plot.figure.tight_layout()
lc_raw_nans_removed_baseline_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {pipeline} NaNs-Removed Raw Baseline Exptime={exptime}s.png")




### Flatten ###
i += 1  # count the step

if flatten:
    # Flatten and plot the light curve
    flatten_window_length = int(flatten_window_proportion * lc_raw_nans_removed.time.shape[0])
    if flatten_window_length % 2 == 0:
        flatten_window_length += 1  # the flatten window length should be an odd number

    lc_flattened, lc_flattened_trend = lc_raw_nans_removed.flatten(window_length=flatten_window_length, polyorder=polyorder, mask=transit_mask_raw_nans_removed, return_trend=True)
    lc_flattened_cdpp = calculate_cdpp(lc_flattened, transit_duration=transit_duration_raw_nans_removed_in_cadence)

    j = 1  # count the sub-step
    lc_flattened_plot, ax_lc_flattened = plt.subplots(figsize=(20, 5))
    lc_flattened.errorbar(ax=ax_lc_flattened, label=f"simplified CDPP={lc_flattened_cdpp:.2f}")
    ax_lc_flattened.set_title(f"{name} Sector {sector} {pipeline} {flatten_window_proportion * 100}% Window Flatten Light Curve Exptime={exptime}s")
    lc_flattened_plot.figure.tight_layout()
    lc_flattened_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {pipeline} {flatten_window_proportion * 100}% Window Flatten Light Curve Exptime={exptime}s.png")

    j += 1  # count the sub-step
    lc_flattened_trend_plot, ax_lc_flattened_trend = plt.subplots(figsize=(20, 5))
    lc_flattened_trend.plot(ax=ax_lc_flattened_trend)
    ax_lc_flattened_trend.set_title(f"{name} Sector {sector} {pipeline} {flatten_window_proportion * 100}% Window Flatten Trend Exptime={exptime}s")
    lc_flattened_trend_plot.figure.tight_layout()
    lc_flattened_trend_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {pipeline} {flatten_window_proportion * 100}% Window Flatten Trend Exptime={exptime}s.png")


    # Retrieve and plot the flattened baseline
    lc_flattened_baseline = lc_flattened[~transit_mask_raw_nans_removed]
    lc_flattened_transit = lc_flattened[transit_mask_raw_nans_removed]
    lc_flattened_baseline_trend = lc_flattened_trend[~transit_mask_raw_nans_removed]
    lc_flattened_transit_trend = lc_flattened_trend[transit_mask_raw_nans_removed]
    lc_flattened_baseline_cdpp = calculate_cdpp(lc_flattened_baseline, transit_duration=transit_duration_raw_nans_removed_in_cadence)

    j += 1  # count the sub-step
    lc_flattened_baseline_plot, ax_lc_flattened_baseline = plt.subplots(figsize=(20, 5))
    lc_flattened_baseline.errorbar(ax=ax_lc_flattened_baseline, label=f"simplified CDPP={lc_flattened_baseline_cdpp:.2f}")
    ax_lc_flattened_baseline.set_title(f"{name} Sector {sector} {pipeline} {flatten_window_proportion * 100}% Window Flatten Baseline Light Curve Exptime={exptime}s")
    lc_flattened_baseline_plot.figure.tight_layout()
    lc_flattened_baseline_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {pipeline} {flatten_window_proportion * 100}% Window Flatten Baseline Light Curve Exptime={exptime}s.png")

    j += 1  # count the sub-step
    lc_flattened_baseline_trend_plot, ax_lc_flattened_baseline_trend = plt.subplots(figsize=(20, 5))
    lc_flattened_baseline_trend.plot(ax=ax_lc_flattened_baseline_trend)
    ax_lc_flattened_baseline_trend.set_title(f"{name} Sector {sector} {pipeline} {flatten_window_proportion * 100}% Window Flatten Baseline Trend Exptime={exptime}s")
    lc_flattened_baseline_trend_plot.figure.tight_layout()
    lc_flattened_baseline_trend_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {pipeline} {flatten_window_proportion * 100}% Window Flatten Baseline Trend Exptime={exptime}s.png")


else:
    lc_flattened_baseline = lc_raw_nans_removed_baseline.copy()
    lc_flattened_baseline_cdpp = calculate_cdpp(lc_flattened_baseline, transit_duration=transit_duration_raw_nans_removed_in_cadence)

    lc_flattened_transit = lc_raw_nans_removed_transit.copy()

    lc_flattened = lc_raw_nans_removed.copy()
    lc_flattened_cdpp = calculate_cdpp(lc_flattened)




### Clip ###
i += 1  # count the step


if clip:
    # Clip and plot the baseline
    j = 1 # count the sub-step
    lc_clipped_baseline = lc_flattened_baseline.remove_outliers(sigma=sigma_baseline)
    lc_clipped_baseline_cdpp = calculate_cdpp(lc_clipped_baseline, transit_duration=transit_duration_raw_nans_removed_in_cadence)

    lc_clipped_baseline_plot, ax_lc_clipped_baseline = plt.subplots(figsize=(20, 5))
    lc_clipped_baseline.errorbar(ax=ax_lc_clipped_baseline, label=f"simplified CDPP={lc_clipped_baseline_cdpp:.2f}")
    ax_lc_clipped_baseline.set_title(f"{name} Sector {sector} {pipeline} {sigma_baseline} Sigma Clipped Baseline Exptime={exptime}s")
    lc_clipped_baseline_plot.figure.tight_layout()
    lc_clipped_baseline_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {pipeline} {sigma_baseline} Sigma Clipped Baseline Exptime={exptime}s.png")


    # Clip and plot the transit if there are upper outliers
    if clip_transit:
        j += 1  # count the sub-step
        lc_flattened_baseline_nanmedian = np.nanmedian(lc_flattened_baseline.flux)
        lc_clipped_transit = lc_flattened_transit.remove_outliers(cenfunc=lambda x, **kwargs: lc_flattened_baseline_nanmedian, sigma_lower=sigma_lower_transit, sigma_upper=sigma_upper_transit)  # remove the upper outliers in the transit based on the nanmedian of lc_flattened_baseline

        lc_clipped_transit_plot, ax_lc_clipped_transit = plt.subplots(figsize=(20, 5))
        lc_clipped_transit.errorbar(ax=ax_lc_clipped_transit)
        ax_lc_clipped_transit.set_title(f"{name} Sector {sector} {pipeline} {sigma_upper_transit} Upper Sigma Clipped Transit Exptime={exptime}s")
        lc_clipped_transit_plot.figure.tight_layout()
        lc_clipped_transit_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {pipeline} {sigma_upper_transit} Upper Sigma Clipped Transit Exptime={exptime}s.png")

    else:
        lc_clipped_transit = lc_flattened_transit.copy()


    # Retrieve and plot the clipped light curve
    j += 1 # count the sub-step
    lc_clipped = lc_clipped_baseline.append(lc_clipped_transit)  # append the in-transit NaNs-removed raw light curve to the clipped baseline to retrieve the clipped light curve
    lc_clipped_cdpp = calculate_cdpp(lc_clipped, transit_duration=transit_duration_raw_nans_removed_in_cadence)

    lc_clipped_plot, ax_lc_clipped = plt.subplots(figsize=(20, 5))
    lc_clipped.errorbar(ax=ax_lc_clipped, label=f"simplified CDPP={lc_clipped_cdpp:.2f}")
    ax_lc_clipped.set_title(f"{name} Sector {sector} {pipeline} {sigma_baseline} Sigma Clipped Light Curve Exptime={exptime}s")
    lc_clipped_plot.figure.tight_layout()
    lc_clipped_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {pipeline} {sigma_baseline} Sigma Clipped Light Curve Exptime={exptime}s.png")


else:
    lc_clipped_baseline = lc_flattened_baseline.copy()
    lc_clipped_baseline_cdpp = calculate_cdpp(lc_clipped_baseline, transit_duration=transit_duration_raw_nans_removed_in_cadence)

    lc_clipped_transit = lc_flattened_transit.copy()

    lc_clipped = lc_flattened.copy()
    lc_clipped_cdpp = calculate_cdpp(lc_clipped, transit_duration=transit_duration_raw_nans_removed_in_cadence)




# Wrtie the corrected LightCurve into a .fits file
if flatten and clip:
    correction = "Flattened & Clipped"
elif flatten and not clip:
    correction = "Flattened"
elif not flatten and clip:
    correction = "Clipped"
else:
    correction = "Raw"


lc_clipped.to_fits(eleanor_root_targetdata_source + f"/{name}_Sector {sector}_{method}_{pipeline}_{correction}_Exptime={exptime}s.fits", overwrite=True)




### Fold ###
# Fold the lightcurve based on the fitted parameters and plot the folded light curve
i += 1 # count the step


lc_folded = lc_clipped.fold(period=period_raw_nans_removed, epoch_time=epoch_time_raw_nans_removed)
lc_folded_cdpp = calculate_cdpp(lc_folded, transit_duration=transit_duration_raw_nans_removed_in_cadence)

lc_folded_plot, ax_lc_folded = plt.subplots(figsize=(10, 5))
lc_folded.errorbar(ax=ax_lc_folded, label=f"simplified CDPP={lc_folded_cdpp:.2f}")
ax_lc_folded.set_title(f"{name} Sector {sector} {pipeline} Period={period_raw_nans_removed}d Folded Light Curve Exptime={exptime}s")
lc_folded_plot.figure.tight_layout()
lc_folded_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02} {name} Sector {sector} {pipeline} Period={period_raw_nans_removed}d Folded Light Curve Exptime={exptime}s.png")




### Bin ###
# Bin the lightcurve and plot the binned light curve
i += 1 # count the step


if bin:
    time_bin_size = exptime * u.second * cadence_bin_size

    lc_binned = lc_folded.bin(time_bin_size=time_bin_size)
    lc_binned_cdpp = calculate_cdpp(lc_binned, transit_duration=transit_duration_raw_nans_removed_in_cadence)

    lc_binned_plot, ax_lc_binned = plt.subplots(figsize=(10, 5))
    # lc_binned.plot(ax=ax_lc_binned, label=f"simplified CDPP={lc_binned_cdpp:.2f}")
    lc_binned.errorbar(ax=ax_lc_binned, label=f"simplified CDPP={lc_binned_cdpp:.2f}")
    ax_lc_binned.set_title(f"{name} Sector {sector} {pipeline} Cadence_Bin_Size={cadence_bin_size} Binned Light Curve Exptime={exptime}s")
    lc_binned_plot.figure.tight_layout()
    lc_binned_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02} {name} Sector {sector} {pipeline} Cadence_Bin_Size={cadence_bin_size} Binned Light Curve Exptime={exptime}s.png")


else:
    lc_binned = lc_folded.copy()
    lc_binned_cdpp = calculate_cdpp(lc_binned, transit_duration=transit_duration_raw_nans_removed_in_cadence)




### ------ Documentation ------ ###
# Print the methodologies and results
methodology_result_file = open(processed_lightcurve_plots_exptime_parent_dir + f"/{name} Sector {sector} {method} Exptime={exptime}s Methodologies And Results.txt", "w", encoding='utf-8')
methodology_result_file.write(f"{name} Sector {sector} {method} Exptime={exptime}s Methodologies And Results\n\n")


methodology_result_file.write("NASA Exoplanet Archive Source Planetary Parameters: \n"
                                    f"Period (Days): {period_NASA}\n"
                                    f"Epoch Time (BJD): {epoch_time_NASA}\n"
                                    f"Transit Duration (Days): {transit_duration_NASA}\n"
                                    f"Transit Depth: {transit_depth_NASA}\n\n")


methodology_result_file.write("Lightkurve: \n\n")


methodology_result_file.write(      f"Raw Light Curve CDPP: {lc_raw_cdpp}\n"
                                    f"NaNs-Removed Raw Baseline CDPP: {lc_raw_nans_removed_baseline_cdpp}\n"
                                    f"NaNs-Removed Raw Light Curve CDPP: {lc_raw_nans_removed_cdpp}\n\n")


methodology_result_file.write(      f"Box Least Squares (BLS) Fitted NaNs-Removed Raw Light Curve Parameters: \n"
                                    f"Period (Days): {period_raw_nans_removed}\n"
                                    f"Epoch Time (BTJD): {epoch_time_raw_nans_removed}\n"
                                    f"Transit Duration (Days): {transit_duration_raw_nans_removed}\n"
                                    f"Transit Duration (Cadences): {transit_duration_raw_nans_removed_in_cadence}\n"
                                    f"Transit Depth: {transit_depth_raw_nans_removed}\n"
                                    f"Fitted the {name} Sector {sector} {pipeline} NaNs-Removed Raw Light Curve Exptime={exptime}s using the Box Least Squares (BLS) method in {pg_bls_raw_nans_removed_fitting_time} seconds.\n\n")


methodology_result_file.write(      f"Flatten: {flatten}\n")
if flatten:
    methodology_result_file.write(f"Flatten Window Proportion: {flatten_window_proportion}\n"
                                  f"Flatten Window Length: {flatten_window_length}\n"
                                  f"Flatten Polyorder: {polyorder}\n"
                                  f"Flattened Baseline CDPP: {lc_flattened_baseline_cdpp}\n"
                                  f"Flattened Light Curve CDPP: {lc_flattened_cdpp}\n\n")
else:
    methodology_result_file.write("\n") # write an empty line


methodology_result_file.write(      f"Clip: {clip}\n")
if clip:
    methodology_result_file.write(f"Baseline Sigma: {sigma_baseline}\n"
                                  f"Transit Upper Sigma: {sigma_upper_transit}\n"
                                  f"Transit Lower Sigma: {sigma_lower_transit}\n"
                                  f"Clipped Baseline CDPP: {lc_clipped_baseline_cdpp}\n"
                                  f"Clipped Light Curve CDPP: {lc_clipped_cdpp}\n\n")
else:
    methodology_result_file.write("\n") # write an empty line


methodology_result_file.write(      f"Folded Period (Days): {period_raw_nans_removed}\n"
                                    f"Folded Epoch Time (BTJD): {epoch_time_raw_nans_removed}\n"
                                    f"Folded Light Curve CDPP: {lc_folded_cdpp}\n\n")


methodology_result_file.write(      f"Bin: {bin}\n")
if bin:
    methodology_result_file.write(f"Cadence Bin Size: {cadence_bin_size}\n"
                                  f"Time Bin Size: {time_bin_size}\n"
                                  f"Binned Light Curve CDPP: {lc_binned_cdpp}\n\n")
else:
    methodology_result_file.write("\n") # write an empty line


methodology_result_file.close()