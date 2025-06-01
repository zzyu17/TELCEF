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

flatten = True
flatten_window_proportion = 0.02

clip = False
sigma = 3

bin = True
cadence_bin_size = 2.5




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




### ------ Lightkurve ------ ###
### Raw ###
# Download and plot the raw light curve
i = 1 # count the step

j = 1 # count the sub-step
lc_raw = lk.search_lightcurve(name, sector=sector, author=f'*{pipeline}*', exptime=exptime).download()
print(f"Downloaded and processing {name} Sector {sector} {pipeline} Exptime={exptime}s Light Curve...\n")
lc_raw_cdpp = lc_raw.estimate_cdpp()

lc_raw_plot, ax_lc_raw = plt.subplots(figsize=(20, 5))
lc_raw.errorbar(ax=ax_lc_raw, label=f"simplified CDPP={lc_raw_cdpp:.2f}")
ax_lc_raw.set_title(f"{name} Sector {sector} {pipeline} Raw Light Curve Exptime={exptime}s")
lc_raw_plot.figure.tight_layout()
lc_raw_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {pipeline} Raw Light Curve Exptime={exptime}s.png")


j += 1 # count the sub-step
lc_raw_remove_nans = lc_raw.remove_nans() # Remove NaNs from the raw light curve
lc_raw_remove_nans_cdpp = lc_raw_remove_nans.estimate_cdpp()

lc_raw_remove_nans_plot, ax_lc_raw_remove_nans = plt.subplots(figsize=(20, 5))
lc_raw_remove_nans.errorbar(ax=ax_lc_raw_remove_nans, label=f"simplified CDPP={lc_raw_remove_nans_cdpp:.2f}")
ax_lc_raw_remove_nans.set_title(f"{name} Sector {sector} {pipeline} NaNs-Removed Raw Light Curve Exptime={exptime}s")
lc_raw_remove_nans_plot.figure.tight_layout()
lc_raw_remove_nans_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {pipeline} NaNs-Removed Raw Light Curve Exptime={exptime}s.png")


# Fit the NaNs-removed raw light curve using the Box Least Squares (BLS) method
j += 1 # count the sub-step
pg_bls_raw_remove_nans_start_time = time.time()  # measure the start time
period_raw_remove_nans_range = np.linspace(0.5, 27, 10000) ##### set the range of period to search #####
pg_bls_raw_remove_nans = lc_raw_remove_nans.to_periodogram(method='bls', period=period_raw_remove_nans_range, frequency_factor=500)
pg_bls_raw_remove_nans_end_time = time.time()  # measure the end time
pg_bls_raw_remove_nans_fitting_time = pg_bls_raw_remove_nans_end_time - pg_bls_raw_remove_nans_start_time # calculate the fitting time
print(f"Fitted the {name} Sector {sector} {pipeline} NaNs-Removed Raw Light Curve Exptime={exptime}s using the Box Least Squares (BLS) method in {pg_bls_raw_remove_nans_fitting_time} seconds.\n")

pg_bls_raw_remove_nans_plot, ax_pg_bls_raw_remove_nans = plt.subplots(figsize=(20, 5))
pg_bls_raw_remove_nans.plot(ax=ax_pg_bls_raw_remove_nans)
ax_pg_bls_raw_remove_nans.set_title(f"{name} Sector {sector} {pipeline} NaNs-Removed Raw BLS Periodogram Exptime={exptime}s")
pg_bls_raw_remove_nans_plot.figure.tight_layout()
pg_bls_raw_remove_nans_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {pipeline} NaNs-Removed Raw BLS Periodogram Exptime={exptime}s.png")


j += 1 # count the sub-step
period_raw_remove_nans = pg_bls_raw_remove_nans.period_at_max_power.value
epoch_time_raw_remove_nans = pg_bls_raw_remove_nans.transit_time_at_max_power.value
transit_duration_raw_remove_nans = pg_bls_raw_remove_nans.duration_at_max_power.value
transit_duration_raw_remove_nans_in_cadence = int(round(transit_duration_raw_remove_nans / exptime_in_days))  # convert the transit duration to number of cadences
transit_depth_raw_remove_nans = pg_bls_raw_remove_nans.depth_at_max_power.value / np.nanmedian(lc_raw_remove_nans.flux.value) # calculate the normalized transit depth
transit_mask_raw_remove_nans = pg_bls_raw_remove_nans.get_transit_mask(period=period_raw_remove_nans, transit_time=epoch_time_raw_remove_nans, duration=transit_duration_raw_remove_nans)
lc_bls_transit_model_raw_remove_nans = pg_bls_raw_remove_nans.get_transit_model(period=period_raw_remove_nans, transit_time=epoch_time_raw_remove_nans, duration=transit_duration_raw_remove_nans)

lc_bls_transit_model_raw_remove_nans_plot, ax_lc_bls_transit_model_raw_remove_nans = plt.subplots(figsize=(20, 5))
lc_raw_remove_nans.scatter(ax=ax_lc_bls_transit_model_raw_remove_nans, label=f"NaNs-Removed Raw Light Curve Exptime={exptime}s")
lc_bls_transit_model_raw_remove_nans.plot(ax=ax_lc_bls_transit_model_raw_remove_nans, c='red', label=f"NaNs-Removed Raw BLS Transit Model Exptime={exptime}s")
ax_lc_bls_transit_model_raw_remove_nans.set_title(f"{name} Sector {sector} {pipeline} NaNs-Removed Raw Light Curve & BLS Transit Model Exptime={exptime}s")
lc_bls_transit_model_raw_remove_nans_plot.figure.tight_layout()
lc_bls_transit_model_raw_remove_nans_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {pipeline} NaNs-Removed Raw Light Curve & BLS Transit Model Exptime={exptime}s.png")


lc_raw_remove_nans_baseline = lc_raw_remove_nans[~transit_mask_raw_remove_nans]  # get the out-of-transit light curve (baseline)




### Flatten ###
# Flatten the lightcurve and plot the flatten light curve
i += 1 # count the step


if flatten:
    flatten_window_length = int(lc_raw.time.shape[0] * flatten_window_proportion)
    j = 1  # count the sub-step
    if flatten_window_length % 2 == 0:
        flatten_window_length += 1  # the window length should be an odd number

    lc_flatten = lc_raw_remove_nans.flatten(window_length=flatten_window_length)
    lc_flatten_cdpp = lc_flatten.estimate_cdpp()
    lc_flattened, lc_flattened_trend = lc_raw_remove_nans.flatten(window_length=flatten_window_length, return_trend=True)

    lc_flattened_plot, ax_lc_flattened = plt.subplots(figsize=(20, 5))
    lc_flattened.errorbar(ax=ax_lc_flattened, label=f"simplified CDPP={lc_flattened_cdpp:.2f}")
    ax_lc_flattened.set_title(f"{name} Sector {sector} {pipeline} {flatten_window_proportion_initial * 100}% Window Flatten Light Curve Exptime={exptime}s")
    lc_flattened_plot.figure.tight_layout()
    lc_flattened_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {pipeline} {flatten_window_proportion_initial * 100}% Window Flatten Light Curve Exptime={exptime}s.png")

    j += 1  # count the sub-step
    lc_flattened_trend_plot, ax_lc_flattened_trend = plt.subplots(figsize=(20, 5))
    lc_flattened_trend.plot(ax=ax_lc_flattened_trend)
    ax_lc_flattened_trend.set_title(f"{name} Sector {sector} {pipeline} {flatten_window_proportion_initial * 100}% Window Flatten Trend Exptime={exptime}s")
    lc_flattened_trend_plot.figure.tight_layout()
    lc_flattened_trend_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {pipeline} {flatten_window_proportion_initial * 100}% Window Flatten Trend Exptime={exptime}s.png")


else:
    lc_flatten = lc_raw_remove_nans
    lc_flatten_cdpp = lc_flatten.estimate_cdpp()
    lc_flattened = lc_raw_remove_nans.copy()




### Clip ###
# Clip the outliers and plot the clipped light curve
i += 1 # count the step


if clip:
    lc_clipped = lc_flatten.remove_outliers(sigma=sigma)
    lc_clipped_cdpp = lc_clipped.estimate_cdpp()
    lc_clipped = lc_flattened.remove_outliers(sigma=sigma_lower_limit)

    lc_clipped_plot, ax_lc_clipped = plt.subplots(figsize=(20, 5))
    lc_clipped.errorbar(ax=ax_lc_clipped, label=f"simplified CDPP={lc_clipped_cdpp:.2f}")
    ax_lc_clipped.set_title(f"{name} Sector {sector} {pipeline} {sigma_lower_limit} Sigma Clipped Light Curve Exptime={exptime}s")
    lc_clipped_plot.figure.tight_layout()
    lc_clipped_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02} {name} Sector {sector} {pipeline} {sigma_lower_limit} Sigma Clipped Light Curve Exptime={exptime}s.png")


else:
    lc_clipped = lc_flatten
    lc_clipped_cdpp = lc_clipped.estimate_cdpp()
    lc_clipped = lc_flattened.copy()




### Fold ###
# Fold the lightcurve based on the fitted parameters and plot the folded light curve
i += 1 # count the step

# Retrieve the epoch time
epoch_time_NASA_TESS = epoch_time_NASA - TESS_time
epoch_time = epoch_time_NASA_TESS
n = 0
while True:
    n += 1
    epoch_time += period_NASA
    if epoch_time >= lc_clipped.time.value[0]:
        break
differences = np.abs(lc_clipped.time.value - epoch_time)
min_diff_index = np.argmin(differences)
epoch_time_TESS = lc_clipped.time.value[min_diff_index]

lc_folded = lc_clipped.fold(period=period, epoch_time=epoch_time_obs)
lc_folded_cdpp = lc_folded.estimate_cdpp()
lc_folded = lc_clipped.fold(period=period_NASA, epoch_time=epoch_time_TESS)

lc_folded_plot, ax_lc_folded = plt.subplots(figsize=(10, 5))
lc_folded.errorbar(ax=ax_lc_folded, label=f"simplified CDPP={lc_folded_cdpp:.2f}")
ax_lc_folded.set_title(f"{name} Sector {sector} {pipeline} Period={period_NASA}d Folded Light Curve Exptime={exptime}s")
lc_folded_plot.figure.tight_layout()
lc_folded_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02} {name} Sector {sector} {pipeline} Period={period_NASA}d Folded Light Curve Exptime={exptime}s.png")




### Bin ###
# Bin the lightcurve and plot the binned light curve
i += 1 # count the step


if bin:
    time_bin_size = exptime * u.second * cadence_bin_size

    lc_binned = lc_folded.bin(time_bin_size=time_bin_size)
    lc_binned_cdpp = lc_binned.estimate_cdpp()

    lc_binned_plot, ax_lc_binned = plt.subplots(figsize=(10, 5))
    lc_binned.plot(ax=ax_lc_binned, label=f"simplified CDPP={lc_binned_cdpp:.2f}")
    lc_binned.errorbar(ax=ax_lc_binned, label=f"simplified CDPP={lc_binned_cdpp:.2f}")
    ax_lc_binned.set_title(f"{name} Sector {sector} {pipeline} Cadence_Bin_Size={cadence_bin_size} Binned Light Curve Exptime={exptime}s")
    lc_binned_plot.figure.tight_layout()
    lc_binned_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02} {name} Sector {sector} {pipeline} Cadence_Bin_Size={cadence_bin_size} Binned Light Curve Exptime={exptime}s.png")


else:
    lc_binned = lc_folded
    lc_binned_cdpp = lc_binned.estimate_cdpp()
    lc_binned = lc_folded.copy()




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

methodology_result_file.write(      f"Raw Light Curve Simplified CDPP: {lc_raw_cdpp}\n"
                                    f"NaNs-Removed Raw Light Curve Simplified CDPP: {lc_raw_remove_nans_cdpp}\n"
                                    f"BLS Fitted NaNs-Removed Raw Light Curve Period (Days): {period_raw_remove_nans}\n"
                                    f"BLS Fitted NaNs-Removed Raw Light Curve Epoch Time (BTJD): {epoch_time_raw_remove_nans}\n"
                                    f"BLS Fitted NaNs-Removed Raw Light Curve Transit Duration (Days): {transit_duration_raw_remove_nans}\n"
                                    f"BLS Fitted NaNs-Removed Raw Light Curve Transit Duration (Cadences): {transit_duration_raw_remove_nans_in_cadence}\n"
                                    f"BLS Fitted NaNs-Removed Raw Light Curve Transit Depth: {transit_depth_raw_remove_nans}\n"
                                    f"Fitted the {name} Sector {sector} {pipeline} NaNs-Removed Raw Light Curve Exptime={exptime}s using the Box Least Squares (BLS) method in {pg_bls_raw_remove_nans_fitting_time} seconds.\n\n")

methodology_result_file.write(      f"Flatten: {flatten}\n")
if flatten:
    methodology_result_file.write(f"Flatten Window Proportion: {flatten_window_proportion_initial}\n"
                                  f"Flatten Light Curve Simplified CDPP: {lc_flattened_cdpp}\n\n")
else:
    methodology_result_file.write("\n") # write an empty line

methodology_result_file.write(      f"Clip: {clip}\n")
if clip:
    methodology_result_file.write(f"Sigma: {sigma_lower_limit}\n"
                                  f"Clipped Light Curve Simplified CDPP: {lc_clipped_cdpp}\n\n")
else:
    methodology_result_file.write("\n") # write an empty line

methodology_result_file.write(      f"Folded Period (Days): {period_NASA}\n"
                                    f"Folded Epoch Time (BTJD): {epoch_time_TESS}\n"
                                    f"Folded Light Curve Simplified CDPP: {lc_folded_cdpp}\n\n")

methodology_result_file.write(      f"Bin: {bin}\n")
if bin:
    methodology_result_file.write(f"Cadence Bin Size: {cadence_bin_size}\n")
    methodology_result_file.write(f"Binned Light Curve Simplified CDPP: {lc_binned_cdpp}\n\n")
else:
    methodology_result_file.write("\n") # write an empty line

methodology_result_file.close()