import lightkurve as lk
import os
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u


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

# ##### Define the source parameters #####
# name = "WASP-107"
# tic = 429302040
# coords = (188.38685041666665, -10.146173611111111)
# gaia = 3578638842054261248
# tess_mag = 10.418

##### Define the Lightkurve parameters depending on the exptime #####
exptime = 20

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

lc_raw_plot, ax_raw = plt.subplots(figsize=(20, 5))
lc_raw.errorbar(ax=ax_raw, label=f"CDPP={lc_raw_cdpp:.2f}")
ax_raw.set_title(f"{name} Sector {sector} {pipeline} Raw Light Curve Exptime={exptime}s")
lc_raw_plot.figure.tight_layout()
lc_raw_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {pipeline} Raw Light Curve Exptime={exptime}s.png")


j += 1 # count the sub-step
lc_raw_remove_nans = lc_raw.remove_nans() # Remove NaNs from the raw light curve
lc_raw_remove_nans_cdpp = lc_raw_remove_nans.estimate_cdpp()

lc_raw_remove_nans_plot, ax_raw_remove_nans = plt.subplots(figsize=(20, 5))
lc_raw_remove_nans.errorbar(ax=ax_raw_remove_nans, label=f"CDPP={lc_raw_remove_nans_cdpp:.2f}")
ax_raw_remove_nans.set_title(f"{name} Sector {sector} {pipeline} NaNs-Removed Raw Light Curve Exptime={exptime}s")
lc_raw_remove_nans_plot.figure.tight_layout()
lc_raw_remove_nans_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {pipeline} NaNs-Removed Raw Light Curve Exptime={exptime}s.png")




### Flatten ###
# Flatten the lightcurve and plot the flatten light curve
i += 1 # count the step


if flatten:
    flatten_window_length = int(lc_raw.time.shape[0] * flatten_window_proportion)
    if flatten_window_length % 2 == 0:
        flatten_window_length += 1  # the window length should be an odd number

    lc_flatten = lc_raw_remove_nans.flatten(window_length=flatten_window_length)
    lc_flatten_cdpp = lc_flatten.estimate_cdpp()

    lc_flatten_plot, ax_flatten = plt.subplots(figsize=(20, 5))
    lc_flatten.errorbar(ax=ax_flatten, label=f"CDPP={lc_flatten_cdpp:.2f}")
    ax_flatten.set_title(f"{name} Sector {sector} {pipeline} {flatten_window_proportion * 100}% Window Flatten Light Curve Exptime={exptime}s")
    lc_flatten_plot.figure.tight_layout()
    lc_flatten_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02} {name} Sector {sector} {pipeline} {flatten_window_proportion * 100}% Window Flatten Light Curve Exptime={exptime}s.png")


else:
    lc_flatten = lc_raw_remove_nans
    lc_flatten_cdpp = lc_flatten.estimate_cdpp()




### Clip ###
# Clip the outliers and plot the clipped light curve
i += 1 # count the step


if clip:
    lc_clipped = lc_flatten.remove_outliers(sigma=sigma)
    lc_clipped_cdpp = lc_clipped.estimate_cdpp()

    lc_clipped_plot, ax_clipped = plt.subplots(figsize=(20, 5))
    lc_clipped.errorbar(ax=ax_clipped, label=f"CDPP={lc_clipped_cdpp:.2f}")
    ax_clipped.set_title(f"{name} Sector {sector} {pipeline} {sigma} Sigma Clipped Light Curve Exptime={exptime}s")
    lc_clipped_plot.figure.tight_layout()
    lc_clipped_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02} {name} Sector {sector} {pipeline} {sigma} Sigma Clipped Light Curve Exptime={exptime}s.png")


else:
    lc_clipped = lc_flatten
    lc_clipped_cdpp = lc_clipped.estimate_cdpp()




### Fold ###
# Fold the lightcurve based on the fitted parameters and plot the folded light curve
i += 1 # count the step

period = 3.067853 # From NASA Exoplanet Archive

# Retrieve the epoch time
primary_transit_time = 2456125.417574 - TESS_time # BTJD, from NASA Exoplanet Archive
epoch_time = primary_transit_time
n = 0
while True:
    n += 1
    epoch_time += period
    if epoch_time >= lc_clipped.time.value[0]:
        break
differences = np.abs(lc_clipped.time.value - epoch_time)
min_diff_index = np.argmin(differences)
epoch_time_obs = lc_clipped.time.value[min_diff_index]

lc_folded = lc_clipped.fold(period=period, epoch_time=epoch_time_obs)
lc_folded_cdpp = lc_folded.estimate_cdpp()

lc_folded_plot, ax_folded = plt.subplots(figsize=(10, 5))
lc_folded.errorbar(ax=ax_folded, label=f"CDPP={lc_folded_cdpp:.2f}")
ax_folded.set_title(f"{name} Sector {sector} {pipeline} Period={period}d Folded Light Curve Exptime={exptime}s")
lc_folded_plot.figure.tight_layout()
lc_folded_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02} {name} Sector {sector} {pipeline} Period={period}d Folded Light Curve Exptime={exptime}s.png")




### Bin ###
# Bin the lightcurve and plot the binned light curve
i += 1 # count the step


if bin:
    time_bin_size = exptime * u.second * cadence_bin_size

    lc_binned = lc_folded.bin(time_bin_size=time_bin_size)
    lc_binned_cdpp = lc_binned.estimate_cdpp()

    lc_binned_plot, ax_binned = plt.subplots(figsize=(10, 5))
    lc_binned.plot(ax=ax_binned, label=f"CDPP={lc_binned_cdpp:.2f}")
    lc_binned.errorbar(ax=ax_binned, label=f"CDPP={lc_binned_cdpp:.2f}")
    ax_binned.set_title(f"{name} Sector {sector} {pipeline} Cadence_Bin_Size={cadence_bin_size} Binned Light Curve Exptime={exptime}s")
    lc_binned_plot.figure.tight_layout()
    lc_binned_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02} {name} Sector {sector} {pipeline} Cadence_Bin_Size={cadence_bin_size} Binned Light Curve Exptime={exptime}s.png")


else:
    lc_binned = lc_folded
    lc_binned_cdpp = lc_binned.estimate_cdpp()




### ------ Documentation ------ ###
# Print the methodologies and results
methodology_result_file = open(processed_lightcurve_plots_exptime_parent_dir + f"/{name} Sector {sector} {method} Exptime={exptime}s Methodologies And Results.txt", "w", encoding='utf-8')
methodology_result_file.write(f"{name} Sector {sector} {method} Exptime={exptime}s Methodologies And Results\n\n")
methodology_result_file.write("Lightkurve: \n"
                                    f"Raw Light Curve CDPP: {lc_raw_cdpp}\n")


methodology_result_file.write(      f"Flatten: {flatten}\n")
if flatten:
    methodology_result_file.write(f"Flatten Window Proportion: {flatten_window_proportion}\n"
                                  f"Flatten Light Curve CDPP: {lc_flatten_cdpp}\n")


methodology_result_file.write(      f"Clip: {clip}\n")
if clip:
    methodology_result_file.write(f"Sigma: {sigma}\n"
                                  f"Clipped Light Curve CDPP: {lc_clipped_cdpp}\n")


methodology_result_file.write(      f"Folded Period: {period}\n"
                                    f"Folded Epoch Time: {epoch_time_obs}\n"
                                    f"Folded Light Curve CDPP: {lc_folded_cdpp}\n")


methodology_result_file.write(      f"Bin: {bin}\n")
if bin:
    methodology_result_file.write(f"Cadence Bin Size: {cadence_bin_size}\n")
    methodology_result_file.write(f"Binned Light Curve CDPP: {lc_binned_cdpp}\n\n")


methodology_result_file.close()