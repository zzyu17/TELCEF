import os
import time

import astropy.units as u
import lightkurve as lk
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import corner
import numpy as np
from pytransit import QuadraticModel, QuadraticModelCL, RoadRunnerModel, QPower2Model, GeneralModel
import emcee




### ------ Preparations ------ ###
# Set the single sector and author to be processed
sector = 54 ##### set the single sector to be processed #####
author = 'SPOC' ##### set the author to be processed #####

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
while epoch_time_NASA < TESS_time :
    epoch_time_NASA += period_NASA
epoch_time_NASA -= TESS_time # BTJD
transit_duration_NASA = (2.1310 * u.hour).to(u.day).value  # hours to days
transit_depth_NASA = (0.17137) ** 2


##### Define the Lightkurve parameters depending on the exptime #####
exptime = 20
exptime_in_day = (exptime * u.second).to(u.day).value  # convert exposure time to days
alpha_exptime = 0.1 # the plotting alpha coefficient of the light curve corresponding to the exposure time, recommended: 0.1 for 20s, 0.3 for 120s, 0.5 for 600s/1800s

transit_mask_raw_nans_removed_coefficient = 1.8 # set the coefficient of BLS transit mask span manually

flatten = True
flatten_window_proportion = 0.05
flatten_polyorder = 3

clip = True
clip_transit = True
sigma_baseline = 3.0
sigma_upper_transit = 0.4 # set sigma_upper_transit manually
sigma_lower_transit = np.inf # set to np.inf to avoid clipping the real transit




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
processed_lightcurve_plots_parent_dir = processed_lightcurve_plots_dir + f"/{name}_Sector {sector}_{author}"
os.makedirs(processed_lightcurve_plots_parent_dir, exist_ok=True)
processed_lightcurve_plots_exptime_parent_dir = processed_lightcurve_plots_parent_dir + f"/Exposure Time={exptime}s"
os.makedirs(processed_lightcurve_plots_exptime_parent_dir, exist_ok=True)

eleanor_root = os.path.expanduser("~/.eleanor")
eleanor_root_targetdata = eleanor_root + f"/targetdata"
os.makedirs(eleanor_root_targetdata, exist_ok=True)
eleanor_root_targetdata_source = eleanor_root_targetdata + f"/{name}"
os.makedirs(eleanor_root_targetdata_source, exist_ok=True)




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
    cdpp = np.std(mean)
    return cdpp




# Define the sort_lc() function
def sort_lc(lc):
    """
    Ensure the light curve is sorted in strictly increasing time order.
    """
    time = lc.time
    flux = lc.flux
    flux_err = lc.flux_err

    if np.all(np.diff(time.value) > 0):
        return lc

    else:
        # Retrieve the indices that would sort the time array
        sort_indices = np.argsort(time.value)

        sorted_time = time[sort_indices]
        sorted_flux = flux[sort_indices]
        sorted_flux_err = flux_err[sort_indices]

        sorted_lc = lc.copy()
        sorted_lc.time = sorted_time
        sorted_lc.flux = sorted_flux
        sorted_lc.flux_err = sorted_flux_err

        return sorted_lc




### ------ Lightkurve & PyTransit ------ ###
### Raw ###
i = 1 # count the step


# Download and plot the raw light curve
j = 1 # count the sub-step
lc_raw = lk.search_lightcurve(name, sector=sector, author=f'*{author}*', exptime=exptime).download()
print(f"Downloaded and processing {name} Sector {sector} {author} Exptime={exptime}s Light Curve...\n")
lc_raw_cdpp = calculate_cdpp(lc_raw, transit_duration=216) ##### set the transit duration after fitting transit parameters using the BLS method for the first time #####

lc_raw_plot, ax_lc_raw = plt.subplots(figsize=(20, 5))
lc_raw.errorbar(ax=ax_lc_raw, label=f"simplified CDPP={lc_raw_cdpp:.2f}")
ax_lc_raw.set_title(f"{name} Sector {sector} {author} Raw Light Curve Exptime={exptime}s")
lc_raw_plot.figure.tight_layout()
lc_raw_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {author} Raw Light Curve Exptime={exptime}s.png")


j += 1 # count the sub-step
lc_raw_nans_removed = sort_lc(lc_raw.remove_nans()) # remove NaNs from the raw light curve and sort it in strictly increasing time order
lc_raw_nans_removed_cdpp = calculate_cdpp(lc_raw_nans_removed, transit_duration=216) ##### set the transit duration after fitting transit parameters using the BLS method for the first time #####

lc_raw_nans_removed_plot, ax_lc_raw_nans_removed = plt.subplots(figsize=(20, 5))
lc_raw_nans_removed.errorbar(ax=ax_lc_raw_nans_removed, label=f"simplified CDPP={lc_raw_nans_removed_cdpp:.2f}")
ax_lc_raw_nans_removed.set_title(f"{name} Sector {sector} {author} NaNs-Removed Raw Light Curve Exptime={exptime}s")
lc_raw_nans_removed_plot.figure.tight_layout()
lc_raw_nans_removed_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {author} NaNs-Removed Raw Light Curve Exptime={exptime}s.png")


# Fit the NaNs-removed raw light curve using the Box Least Squares (BLS) method
j += 1 # count the sub-step
pg_bls_raw_nans_removed_start_time = time.time() # measure the start time
p_raw_nans_removed_range = np.linspace(0.5, 27, 10000) ##### set the range of period to search #####
pg_bls_raw_nans_removed = lc_raw_nans_removed.to_periodogram(method='bls', period=p_raw_nans_removed_range, frequency_factor=500)
pg_bls_raw_nans_removed_end_time = time.time() # measure the end time
pg_bls_raw_nans_removed_fitting_time = pg_bls_raw_nans_removed_end_time - pg_bls_raw_nans_removed_start_time # calculate the fitting time
print(f"Fitted global parameters of {name} Sector {sector} {author} NaNs-Removed Raw light curve Exptime={exptime}s using the Box Least Squares (BLS) method in {pg_bls_raw_nans_removed_fitting_time} seconds.\n")

pg_bls_raw_nans_removed_plot, ax_pg_bls_raw_nans_removed = plt.subplots(figsize=(20, 5))
pg_bls_raw_nans_removed.plot(ax=ax_pg_bls_raw_nans_removed)
ax_pg_bls_raw_nans_removed.set_title(f"{name} Sector {sector} {author} NaNs-Removed Raw BLS Periodogram Exptime={exptime}s")
pg_bls_raw_nans_removed_plot.figure.tight_layout()
pg_bls_raw_nans_removed_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {author} NaNs-Removed Raw BLS Periodogram Exptime={exptime}s.png")


j += 1 # count the sub-step
p_raw_nans_removed = pg_bls_raw_nans_removed.period_at_max_power.value
t0_raw_nans_removed = pg_bls_raw_nans_removed.transit_time_at_max_power.value
transit_duration_raw_nans_removed = pg_bls_raw_nans_removed.duration_at_max_power.value
transit_duration_in_cadence_raw_nans_removed = int(round(transit_duration_raw_nans_removed / exptime_in_day))  # convert transit duration to number of cadences
transit_depth_raw_nans_removed = pg_bls_raw_nans_removed.depth_at_max_power.value / np.nanmedian(lc_raw_nans_removed.flux.value) # calculate the normalized transit depth
transit_mask_raw_nans_removed = pg_bls_raw_nans_removed.get_transit_mask(period=p_raw_nans_removed, transit_time=t0_raw_nans_removed, duration=transit_duration_raw_nans_removed * transit_mask_raw_nans_removed_coefficient)
lc_bls_transit_model_raw_nans_removed = pg_bls_raw_nans_removed.get_transit_model(period=p_raw_nans_removed, transit_time=t0_raw_nans_removed, duration=transit_duration_raw_nans_removed)

lc_bls_transit_model_raw_nans_removed_plot, ax_lc_bls_transit_model_raw_nans_removed = plt.subplots(figsize=(20, 5))
lc_raw_nans_removed.errorbar(ax=ax_lc_bls_transit_model_raw_nans_removed, label=f"NaNs-Removed Raw Light Curve")
lc_bls_transit_model_raw_nans_removed.plot(ax=ax_lc_bls_transit_model_raw_nans_removed, c='red', label=f"Best Fitted BLS Model")
ax_lc_bls_transit_model_raw_nans_removed.set_title(f"{name} Sector {sector} {author} NaNs-Removed Raw Light Curve & BLS Model Exptime={exptime}s")
lc_bls_transit_model_raw_nans_removed_plot.figure.tight_layout()
lc_bls_transit_model_raw_nans_removed_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {author} NaNs-Removed Raw Light Curve & BLS Model Exptime={exptime}s.png")


# Retrieve and plot Nans-removed raw baseline
j += 1 # count the sub-step
lc_raw_nans_removed_baseline = lc_raw_nans_removed[~transit_mask_raw_nans_removed] # retrieve the out-of-transit light curve (baseline)
lc_raw_nans_removed_transit = lc_raw_nans_removed[transit_mask_raw_nans_removed] # retrieve the in-transit light curve
lc_raw_nans_removed_baseline_cdpp = calculate_cdpp(lc_raw_nans_removed_baseline, transit_duration=transit_duration_in_cadence_raw_nans_removed)

lc_raw_nans_removed_baseline_plot, ax_lc_raw_nans_removed_baseline = plt.subplots(figsize=(20, 5))
lc_raw_nans_removed_baseline.errorbar(ax=ax_lc_raw_nans_removed_baseline, label=f"simplified CDPP={lc_raw_nans_removed_baseline_cdpp:.2f}")
ax_lc_raw_nans_removed_baseline.set_title(f"{name} Sector {sector} {author} NaNs-Removed Raw Baseline Exptime={exptime}s")
lc_raw_nans_removed_baseline_plot.figure.tight_layout()
lc_raw_nans_removed_baseline_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {author} NaNs-Removed Raw Baseline Exptime={exptime}s.png")




### Flatten ###
i += 1  # count the step

if flatten:
    # Flatten and plot the light curve
    flatten_window_length = int(flatten_window_proportion * lc_raw_nans_removed.time.shape[0])
    if flatten_window_length % 2 == 0:
        flatten_window_length += 1  # the flatten window length should be an odd number

    lc_flattened, lc_flattened_trend = lc_raw_nans_removed.flatten(window_length=flatten_window_length, polyorder=flatten_polyorder, mask=transit_mask_raw_nans_removed, return_trend=True)
    lc_flattened_cdpp = calculate_cdpp(lc_flattened, transit_duration=transit_duration_in_cadence_raw_nans_removed)

    j = 1  # count the sub-step
    lc_flattened_plot, ax_lc_flattened = plt.subplots(figsize=(20, 5))
    lc_flattened.errorbar(ax=ax_lc_flattened, label=f"simplified CDPP={lc_flattened_cdpp:.2f}")
    ax_lc_flattened.set_title(f"{name} Sector {sector} {author} {flatten_window_proportion * 100}% Window Flattened Light Curve Exptime={exptime}s")
    lc_flattened_plot.figure.tight_layout()
    lc_flattened_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {author} {flatten_window_proportion * 100}% Window Flattened Light Curve Exptime={exptime}s.png")

    j += 1  # count the sub-step
    lc_flattened_trend_plot, (ax_lc_flattened, ax_lc_flattened_trend) = plt.subplots(2, 1, figsize=(20, 10))
    lc_flattened.errorbar(ax=ax_lc_flattened, label=f"simplified CDPP={lc_flattened_cdpp:.2f}")
    ax_lc_flattened.set_title(f"{name} Sector {sector} {author} {flatten_window_proportion * 100}% Window Flattened Light Curve Exptime={exptime}s")
    lc_flattened_trend.plot(ax=ax_lc_flattened_trend)
    ax_lc_flattened_trend.set_title(f"{name} Sector {sector} {author} {flatten_window_proportion * 100}% Window Flatten Trend Exptime={exptime}s")
    lc_flattened_trend_plot.figure.tight_layout()
    lc_flattened_trend_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {author} {flatten_window_proportion * 100}% Window Flatten Trend Exptime={exptime}s.png")


    # Retrieve and plot the flattened baseline
    lc_flattened_baseline = lc_flattened[~transit_mask_raw_nans_removed]
    lc_flattened_transit = lc_flattened[transit_mask_raw_nans_removed]
    lc_flattened_baseline_trend = lc_flattened_trend[~transit_mask_raw_nans_removed]
    lc_flattened_transit_trend = lc_flattened_trend[transit_mask_raw_nans_removed]
    lc_flattened_baseline_cdpp = calculate_cdpp(lc_flattened_baseline, transit_duration=transit_duration_in_cadence_raw_nans_removed)

    j += 1  # count the sub-step
    lc_flattened_baseline_plot, ax_lc_flattened_baseline = plt.subplots(figsize=(20, 5))
    lc_flattened_baseline.errorbar(ax=ax_lc_flattened_baseline, label=f"simplified CDPP={lc_flattened_baseline_cdpp:.2f}")
    ax_lc_flattened_baseline.set_title(f"{name} Sector {sector} {author} {flatten_window_proportion * 100}% Window Flattened Baseline Light Curve Exptime={exptime}s")
    lc_flattened_baseline_plot.figure.tight_layout()
    lc_flattened_baseline_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {author} {flatten_window_proportion * 100}% Window Flattened Baseline Light Curve Exptime={exptime}s.png")

    j += 1  # count the sub-step
    lc_flattened_baseline_trend_plot, (ax_lc_flattened_baseline, ax_lc_flattened_baseline_trend) = plt.subplots(2, 1, figsize=(20, 10))
    lc_flattened_baseline.errorbar(ax=ax_lc_flattened_baseline, label=f"simplified CDPP={lc_flattened_baseline_cdpp:.2f}")
    ax_lc_flattened_baseline.set_title(f"{name} Sector {sector} {author} {flatten_window_proportion * 100}% Window Flattened Baseline Light Curve Exptime={exptime}s")
    lc_flattened_baseline_trend.plot(ax=ax_lc_flattened_baseline_trend)
    ax_lc_flattened_baseline_trend.set_title(f"{name} Sector {sector} {author} {flatten_window_proportion * 100}% Window Flattened Baseline Trend Exptime={exptime}s")
    lc_flattened_baseline_trend_plot.figure.tight_layout()
    lc_flattened_baseline_trend_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {author} {flatten_window_proportion * 100}% Window Flattened Baseline Trend Exptime={exptime}s.png")


else:
    lc_flattened_baseline = lc_raw_nans_removed_baseline.copy()
    lc_flattened_baseline_cdpp = calculate_cdpp(lc_flattened_baseline, transit_duration=transit_duration_in_cadence_raw_nans_removed)

    lc_flattened_transit = lc_raw_nans_removed_transit.copy()

    lc_flattened = lc_raw_nans_removed.copy()
    lc_flattened_cdpp = calculate_cdpp(lc_flattened)




### Clip ###
i += 1  # count the step


if clip:
    # Clip and plot the baseline
    j = 1 # count the sub-step
    lc_clipped_baseline = lc_flattened_baseline.remove_outliers(sigma=sigma_baseline)
    lc_clipped_baseline_cdpp = calculate_cdpp(lc_clipped_baseline, transit_duration=transit_duration_in_cadence_raw_nans_removed)

    lc_clipped_baseline_plot, ax_lc_clipped_baseline = plt.subplots(figsize=(20, 5))
    lc_clipped_baseline.errorbar(ax=ax_lc_clipped_baseline, label=f"simplified CDPP={lc_clipped_baseline_cdpp:.2f}")
    ax_lc_clipped_baseline.set_title(f"{name} Sector {sector} {author} {sigma_baseline} Sigma Clipped Baseline Exptime={exptime}s")
    lc_clipped_baseline_plot.figure.tight_layout()
    lc_clipped_baseline_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {author} {sigma_baseline} Sigma Clipped Baseline Exptime={exptime}s.png")


    # Clip and plot the transit if there are upper outliers
    if clip_transit:
        j += 1  # count the sub-step
        lc_flattened_baseline_nanmedian = np.nanmedian(lc_flattened_baseline.flux)
        lc_clipped_transit = lc_flattened_transit.remove_outliers(cenfunc=lambda x, **kwargs: lc_flattened_baseline_nanmedian, sigma_lower=sigma_lower_transit, sigma_upper=sigma_upper_transit)  # remove the upper outliers in the transit based on the nanmedian of lc_flattened_baseline

        lc_clipped_transit_plot, ax_lc_clipped_transit = plt.subplots(figsize=(20, 5))
        lc_clipped_transit.errorbar(ax=ax_lc_clipped_transit)
        ax_lc_clipped_transit.set_title(f"{name} Sector {sector} {author} {sigma_upper_transit} Upper Sigma Clipped Transit Exptime={exptime}s")
        lc_clipped_transit_plot.figure.tight_layout()
        lc_clipped_transit_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {author} {sigma_upper_transit} Upper Sigma Clipped Transit Exptime={exptime}s.png")

    else:
        lc_clipped_transit = lc_flattened_transit.copy()


    # Retrieve and plot the clipped light curve
    j += 1 # count the sub-step
    lc_clipped = sort_lc(lc_clipped_baseline.append(lc_clipped_transit))  # append the in-transit NaNs-removed raw light curve to the clipped baseline to retrieve the clipped light curve
    lc_clipped_cdpp = calculate_cdpp(lc_clipped, transit_duration=transit_duration_in_cadence_raw_nans_removed)

    lc_clipped_plot, ax_lc_clipped = plt.subplots(figsize=(20, 5))
    lc_clipped.errorbar(ax=ax_lc_clipped, label=f"simplified CDPP={lc_clipped_cdpp:.2f}")
    ax_lc_clipped.set_title(f"{name} Sector {sector} {author} {sigma_baseline} Sigma Clipped Light Curve Exptime={exptime}s")
    lc_clipped_plot.figure.tight_layout()
    lc_clipped_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {author} {sigma_baseline} Sigma Clipped Light Curve Exptime={exptime}s.png")


else:
    lc_clipped_baseline = lc_flattened_baseline.copy()
    lc_clipped_baseline_cdpp = calculate_cdpp(lc_clipped_baseline, transit_duration=transit_duration_in_cadence_raw_nans_removed)

    lc_clipped_transit = lc_flattened_transit.copy()

    lc_clipped = lc_flattened.copy()
    lc_clipped_cdpp = calculate_cdpp(lc_clipped, transit_duration=transit_duration_in_cadence_raw_nans_removed)




# Write the corrected LightCurve into a .fits file
if flatten and clip:
    correction = "Flattened & Clipped"
elif flatten and not clip:
    correction = "Flattened"
elif not flatten and clip:
    correction = "Clipped"
else:
    correction = "Raw"


##### Assign the corrected light curve #####
lc_corrected = lc_clipped.copy()
lc_corrected_baseline = lc_clipped_baseline.copy()
lc_corrected_transit = lc_clipped_transit.copy()

lc_corrected.to_fits(eleanor_root_targetdata_source + f"/{name}_Sector {sector}_{method}_{author}_{correction}_Exptime={exptime}s.fits", overwrite=True)




##### Define the emcee MCMC sampling parameters #####
fit_global = True
fit_individual = True

max_inter_global = 3 # maximum number of iterations for the global fitting after removing outliers
sigma_global = 3.0 # sigma for the global fitting outlier removal
max_inter_individual = 3 # maximum number of iterations for the individual fitting after removing outliers
sigma_individual = 3.0 # sigma for the individual fitting outlier removal

n_walkers = 32
n_dim_multi = 7 # 7 parameters to be fitted when fitting multiple transits: k, t0, p, a, i, ldc1, ldc2
n_dim_single = 6 # 6 parameters to be fitted when fitting single transit (with period fixed): k, t0, a, i, ldc1, ldc2
n_steps_global = 5000
n_steps_individual = 5000
chain_discard_proportion_global = 0.2
chain_discard_proportion_individual = 0.2 # the proportion of the chain to discard to remove the burn-in phase

individual_transit_check_coefficient = 1.2 # the coefficient of transit duration span to check if the individual light curve contains the transit event

chain_thin_global = 10
chain_thin_individual = 10 # thinning factor of the sample chain when visualizing the process and result
running_mean_window_proportion = 0.05 # window length proportion of the chain to calculate the running mean of the parameters
running_mean_window_length_global = int(running_mean_window_proportion * n_steps_global * (1 - chain_discard_proportion_global) / chain_thin_global)
running_mean_window_length_individual = int(running_mean_window_proportion * n_steps_individual * (1 - chain_discard_proportion_individual) / chain_thin_individual)
individual_transit_plot_coefficient = 2 # the coefficient of the individual transit plot span




# Define the multiple-transit fitting functions
def model_multi(params, transit_model, time=None):
    k, t0, p, a, i, ldc1, ldc2 = params
    tm = transit_model
    if time is not None:
        if isinstance(time, np.ndarray):
            tm.set_data(time=time)
        else:
            tm.set_data(time=time.value)
    model_flux = tm.evaluate(k=k, t0=t0, p=p, a=a, i=i, e=0.0, w=0.0, ldc=[ldc1, ldc2])
    return model_flux

def log_likelihood_multi(params, transit_model, lc):
    model_flux = model_multi(params, transit_model)
    model_log_likelihood = -0.5 * np.sum((lc.flux.value - model_flux)**2 / lc.flux_err.value**2)
    return model_log_likelihood

def log_prior_multi(params, lc):
    k, t0, p, a, i, ldc1, ldc2 = params
    if (0 < k < 1 and np.min(lc.time.value) < t0 < np.max(lc.time.value) and p > 0 and a > 1 and -np.pi/2 < i < np.pi/2 and 0 < ldc1 < 1 and 0 < ldc2 < 1):
        model_log_prior = 0.0
    else:
        model_log_prior = -np.inf
    return model_log_prior

def log_probability_multi(params, transit_model, lc):
    model_log_likelihood = log_likelihood_multi(params, transit_model, lc)
    model_log_prior = log_prior_multi(params, lc)
    model_log_probability = model_log_likelihood + model_log_prior
    if np.isfinite(model_log_prior):
        return model_log_probability
    else:
        return -np.inf


# Define the single-transit fitting functions
def model_single(params, period, transit_model, time=None):
    k, t0, a, i, ldc1, ldc2 = params
    tm = transit_model
    if time is not None:
        if isinstance(time, np.ndarray):
            tm.set_data(time=time)
        else:
            tm.set_data(time=time.value)
    model_flux = tm.evaluate(k=k, t0=t0, p=period, a=a, i=i, e=0.0, w=0.0, ldc=[ldc1, ldc2])
    return model_flux

def log_likelihood_single(params, period, transit_model, lc):
    model_flux = model_single(params, period, transit_model)
    model_log_likelihood = -0.5 * np.sum((lc.flux.value - model_flux)**2 / lc.flux_err.value**2)
    return model_log_likelihood

def log_prior_single(params, period, lc):
    k, t0, a, i, ldc1, ldc2 = params
    p =  period
    if (0 < k < 1 and np.min(lc.time.value) < t0 < np.max(lc.time.value) and p > 0 and a > 1 and -np.pi/2 < i < np.pi/2 and 0 < ldc1 < 1 and 0 < ldc2 < 1):
        model_log_prior = 0.0
    else:
        model_log_prior = -np.inf
    return model_log_prior

def log_probability_single(params, period, transit_model, lc):
    model_log_likelihood = log_likelihood_single(params, period, transit_model, lc)
    model_log_prior = log_prior_single(params, period, lc)
    model_log_probability = model_log_likelihood + model_log_prior
    if np.isfinite(model_log_prior):
        return model_log_probability
    else:
        return -np.inf




# Define the calculate_transit_duration() & calculate_transit_depth() functions
def calculate_transit_duration(k, p, a, i):
    """
    Function to calculate the transit duration (t_14) based on the
    normalized planetary radius (k), period (p), normalized semi-major axis (a), and inclination (i) in radians.
    """
    b = a * np.cos(i)
    t_14 = np.arcsin(np.sqrt((1 + k) ** 2 - b ** 2) / (a * np.sin(i))) * p / np.pi

    return t_14


def calculate_transit_depth(k, a, i, ldc, transit_model):
    if transit_model.lower() == 'quadratic':
        tm = QuadraticModel()
    elif transit_model.lower() == 'quadraticcl':
        tm = QuadraticModelCL()
    elif transit_model.lower() == 'roadrunner':
        tm = RoadRunnerModel()
    elif transit_model.lower() == 'qpower2':
        tm = QPower2Model()
    elif transit_model.lower() == 'general':
        tm = GeneralModel()
    else:
        raise ValueError(f"Unsupported transit model: {transit_model}. "
                         f"Supported models are: 'Quadratic', 'QuadraticCL', 'RoadRunner', 'QPower2', 'General'.")

    t0 = 0.0 # t0 should be set to 0.0
    p = 1.0 # p can be set randomly

    tm.set_data(time=[0.0]) # calculate the transit depth at t=0.0 (i.e., at the epoch time)

    flux_out = tm.evaluate(k=0, t0=t0, p=p, a=a, i=i, ldc=ldc) # calculate the out-of-transit flux, where k=0 means no planet
    flux_in = tm.evaluate(k=k, t0=t0, p=p, a=a, i=i, ldc=ldc) # calculate the in-transit flux at the epoch time
    transit_depth = flux_out - flux_in # calculate the transit depth

    return transit_depth




# Define the max_deviation() function to calculate the deviation of the most deviated parameter from the NaN-median of the other parameters
def max_deviation(arr):
    array = np.asarray(arr)
    median_all = np.nanmedian(array)
    abs_deviation = np.abs(array - median_all)
    max_dev_idx = np.nanargmax(abs_deviation)
    max_dev_value = array[max_dev_idx]

    array_clipped = np.delete(array, max_dev_idx)
    median_clipped = np.nanmedian(array_clipped)
    std_clipped = np.nanstd(array_clipped)

    max_dev_n_stds = np.abs(max_dev_value - median_clipped) / std_clipped if std_clipped != 0 else np.inf

    return max_dev_idx, max_dev_value, max_dev_n_stds




### Global Fitting ###
i += 1 # count the step


if fit_global:
    # Initialize the parameters dictionary
    params_global_best_dict = {
        'model': None,
        'k': None, 't0': None, 'p': None, 'a': None, 'i': None, 'i_in_degree': None, 'ldc1': None, 'ldc2': None, 'ldc': None,
        'transit_duration': None, 'transit_duration_in_cadence': None, 'transit_depth': None,
        'n_fitting_iteration': None, 'residual_std': None, 'chi_square': None, 'reduced_chi_square': None
    }

    params_global_name = ['k', 't0', 'p', 'a', 'i', 'ldc1', 'ldc2']


    ##### Initialize and configure the transit model #####
    transit_model_name_global = 'Quadratic'
    transit_model_global = QuadraticModel()
    transit_model_global.set_data(lc_corrected.time.value)


    ##### Set the initial global parameters based on the NaNs-removed raw light curve BLS-fitted parameters #####
    k_global_initial = np.sqrt(transit_depth_raw_nans_removed) # k: normalized planetary radius, i.e., R_p/R_s
    t0_global_initial = t0_raw_nans_removed # t0: epoch time
    p_global_initial = p_raw_nans_removed # p: period
    a_global_initial = 10.0 # a: normalized semi-major axis, i.e., a/R_s
    i_global_initial = np.pi / 2 # i: inclination in radians
    ldc1_global_initial = 0.2 # ldc1: linear coefficient
    ldc2_global_initial = 0.3 # ldc2: quadratic coefficient
    ldc_global_initial = [ldc1_global_initial, ldc2_global_initial] # ldc: quadratic limb darkening coefficients

    params_global_initial = [k_global_initial, t0_global_initial, p_global_initial, a_global_initial, i_global_initial, ldc1_global_initial, ldc2_global_initial]


    # Store the parameters in the dictionary
    params_global_best_dict['model'] = transit_model_name_global


    inter_global = 0
    while inter_global < max_inter_global:
        inter_global += 1
        # Initialize walkers
        params_global_position = params_global_initial + 1e-4 * np.random.randn(n_walkers, n_dim_multi)

        # Initialize and run the MCMC sampler
        params_global_sampler = emcee.EnsembleSampler(n_walkers, n_dim_multi, log_probability_multi, args=(transit_model_global, lc_corrected))
        params_global_sampler.run_mcmc(params_global_position, n_steps_global, progress=True, progress_kwargs={'desc': f"Global Fitting Interation {inter_global}: "})

        # Retrieve the best fitted parameters and their uncertainties from the MCMC sampler
        params_global_samples = params_global_sampler.get_chain(discard=int(n_steps_global * chain_discard_proportion_global), flat=True)
        params_global_best = np.median(params_global_samples, axis=0)
        params_global_best_lower_error = params_global_best - np.percentile(params_global_samples, 16, axis=0)
        params_global_best_upper_error = np.percentile(params_global_samples, 84, axis=0) - params_global_best

        k_global_samples, t0_global_samples, p_global_samples, a_global_samples, i_global_samples, ldc1_global_samples, ldc2_global_samples = params_global_samples.T
        ldc_global_samples = np.column_stack((ldc1_global_samples, ldc2_global_samples))

        k_global_best, t0_global_best, p_global_best, a_global_best, i_global_best, ldc1_global_best, ldc2_global_best = params_global_best
        k_global_best_lower_error, t0_global_best_lower_error, p_global_best_lower_error, a_global_best_lower_error, i_global_best_lower_error, ldc1_global_best_lower_error, ldc2_global_best_lower_error = params_global_best_lower_error
        k_global_best_upper_error, t0_global_best_upper_error, p_global_best_upper_error, a_global_best_upper_error, i_global_best_upper_error, ldc1_global_best_upper_error, ldc2_global_best_upper_error = params_global_best_upper_error

        i_in_degree_global_best, i_in_degree_global_best_lower_error, i_in_degree_global_best_upper_error = np.rad2deg(i_global_best), np.rad2deg(i_global_best_lower_error), np.rad2deg(i_global_best_upper_error)
        ldc_global_best, ldc_global_best_lower_error, ldc_global_best_upper_error = [ldc1_global_best, ldc2_global_best], [ldc1_global_best_lower_error, ldc2_global_best_lower_error], [ldc1_global_best_upper_error, ldc2_global_best_upper_error]

        # calculate the transit duration
        transit_duration_global_samples = [calculate_transit_duration(k, p, a, i) for k, p, a, i in zip(k_global_samples, p_global_samples, a_global_samples, i_global_samples)]
        transit_duration_global_best = np.median(transit_duration_global_samples, axis=0)
        transit_duration_global_best_lower_error = transit_duration_global_best - np.percentile(transit_duration_global_samples, 16, axis=0)
        transit_duration_global_best_upper_error = np.percentile(transit_duration_global_samples, 84, axis=0) - transit_duration_global_best

        # convert transit duration to number of cadences
        transit_duration_in_cadence_global_best = int(round(transit_duration_global_best / exptime_in_day))
        transit_duration_in_cadence_global_best_lower_error = int(round(transit_duration_global_best_lower_error / exptime_in_day))
        transit_duration_in_cadence_global_best_upper_error = int(round(transit_duration_global_best_upper_error / exptime_in_day))

        # calculate the normalized transit depth
        transit_depth_global_samples = [calculate_transit_depth(k, a, i, ldc, transit_model_name_global) for k, a, i, ldc in zip(k_global_samples, a_global_samples, i_global_samples, ldc_global_samples)]
        transit_depth_global_best = np.median(transit_depth_global_samples, axis=0)
        transit_depth_global_best_lower_error = transit_depth_global_best - np.percentile(transit_depth_global_samples, 16, axis=0)
        transit_depth_global_best_upper_error = np.percentile(transit_depth_global_samples, 84, axis=0) - transit_depth_global_best


        # calculate the best fitted model flux
        lc_global_best_fit = lc_corrected.copy()
        lc_global_best_fit.flux = model_multi(params_global_best, transit_model_global, time=lc_global_best_fit.time.value) * lc_corrected.flux.unit
        lc_global_best_fit.flux_err = np.zeros(len(lc_global_best_fit.flux_err))

        # calculate the best fitted model residuals
        lc_global_best_fit_residual = lc_corrected.copy()
        lc_global_best_fit_residual.flux = lc_corrected.flux - lc_global_best_fit.flux
        lc_global_best_fit_residual.flux_err = lc_corrected.flux_err - lc_global_best_fit.flux_err

        # calculate the residual standard deviation
        residual_std_global = np.std(lc_global_best_fit_residual.flux.value)

        # calculate the chi-square and reduced chi-square of the best fitted model
        chi_square_global = np.sum((lc_global_best_fit_residual.flux.value / lc_corrected.flux_err.value) ** 2)
        reduced_chi_square_global = chi_square_global / (len(lc_corrected.flux.value) - n_dim_multi)


        # Plot the visualization plots of the fitting process and result
        # Plot the parameters trace and evolution plots
        j = 1 # count the sub-step

        params_global_samples_thinned_unflattened = params_global_sampler.get_chain(discard=int(n_steps_global * chain_discard_proportion_global), thin=chain_thin_global, flat=False) # retrieve the thinned and unflattened sample chains from the MCMC sampler

        params_global_trace_evolution_plot = plt.figure(figsize=(20, 2 * n_dim_multi))
        params_global_trace_evolution_gs = GridSpec(n_dim_multi, 2, wspace=0.05)

        params_global_moving_means = []
        for d in range(n_dim_multi):
            param_global_mean = np.mean(params_global_samples_thinned_unflattened[:, :, d], axis=1)
            param_global_moving_means = np.convolve(param_global_mean, np.ones(running_mean_window_length_global) / running_mean_window_length_global, mode='valid')
            params_global_moving_means.append(param_global_moving_means)

        for d in range(n_dim_multi):
            # Plot the parameter trace plot on the left side
            ax_trace_global = params_global_trace_evolution_plot.add_subplot(params_global_trace_evolution_gs[d, 0])
            for w in range(n_walkers):
                ax_trace_global.plot(params_global_samples_thinned_unflattened[:, w, d], alpha=0.5, linewidth=0.8)
            ax_trace_global.set_ylabel(params_global_name[d])
            ax_trace_global.grid(True, alpha=0.3)
            # only show the x-axis label on the last subplot
            if d == n_dim_multi - 1:
                ax_trace_global.set_xlabel("Step Number")
            else:
                ax_trace_global.tick_params(labelbottom=False)
            # only show the title on the first subplot
            if d == 0:
                ax_trace_global.set_title("Trace Of Parameters", fontsize='x-large')

            # Plot the parameter evolution plot on the right side
            ax_evolution_global = params_global_trace_evolution_plot.add_subplot(params_global_trace_evolution_gs[d, 1], sharey=ax_trace_global)
            ax_evolution_global.plot(params_global_moving_means[d], c='red')
            ax_evolution_global.tick_params(labelleft=False)  # hide the y-axis labels
            ax_evolution_global.grid(True, alpha=0.3)
            # only show the x-axis label on the last subplot
            if d == n_dim_multi - 1:
                ax_evolution_global.set_xlabel("Step Number")
            else:
                ax_evolution_global.tick_params(labelbottom=False)
            # only show the legend on the first subplot
            if d == 0:
                ax_evolution_global.set_title(f"Evolution ({running_mean_window_proportion * 100}% Window Running Mean) Of Parameters", fontsize='x-large')

        params_global_trace_evolution_plot.suptitle(f"{name} Sector {sector} {author} Global Fitting Trace And Evolution Plot (Thinned By {chain_thin_global}) Exptime={exptime}s Interation {inter_global}", fontsize='xx-large')
        params_global_trace_evolution_plot.figure.subplots_adjust(wspace=0.05)
        params_global_trace_evolution_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01}-{inter_global} {name} Sector {sector} {author} Global Fitting Trace And Evolution Plot Exptime={exptime}s Interation {inter_global}.png")


        # Plot the parameters posterior distribution plot
        j += 1  # count the sub-step

        params_global_corner_plot = corner.corner(params_global_samples, labels=params_global_name, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".4f", figsize=(20, 25))
        params_global_corner_plot.suptitle(f"{name} Sector {sector} {author} Global Fitting Parameters Posterior Distribution Corner Plot Exptime={exptime}s Interation {inter_global}", fontsize='xx-large', y=1.05)
        params_global_corner_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01}-{inter_global} {name} Sector {sector} {author} Global Fitting Parameters Posterior Distribution Corner Plot Exptime={exptime}s Interation {inter_global}.png", bbox_inches='tight')


        # Plot the best fitted model light curve and residuals
        j += 1  # count the sub-step

        lc_global_best_fit_plot, (ax_lc_global_best_fit, ax_lc_global_best_fit_residual) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
        lc_corrected.scatter(ax=ax_lc_global_best_fit, label=None, s=0.1, alpha=1.0)
        lc_corrected.errorbar(ax=ax_lc_global_best_fit, label="Corrected Light Curve", alpha=alpha_exptime)
        lc_global_best_fit.plot(ax=ax_lc_global_best_fit, c='red', label=f"Best Fitted {transit_model_name_global} Model, chi-square={chi_square_global:.2f}, reduced chi-square={reduced_chi_square_global:.2f}")
        ax_lc_global_best_fit.legend(loc='lower right')
        ax_lc_global_best_fit.set_ylabel("Flux")
        ax_lc_global_best_fit.set_xlabel("") # plot the best fitted model flux
        lc_global_best_fit_residual.errorbar(ax=ax_lc_global_best_fit_residual, c='green', label=f"Best Fitted Model Residuals, residual std={residual_std_global:.6f}")
        ax_lc_global_best_fit_residual.legend(loc='upper right')
        ax_lc_global_best_fit_residual.set_ylabel("Residuals") # plot the best fitted model residuals

        # set the x-axis limits and y-axis limits of the latter interation to the same as the first one
        if inter_global == 1:
            ax_lc_global_best_fit_x_lim = ax_lc_global_best_fit.get_xlim()
            ax_lc_global_best_fit_y_lim = ax_lc_global_best_fit.get_ylim()
            ax_lc_global_best_fit_residual_x_lim = ax_lc_global_best_fit_residual.get_xlim()
            ax_lc_global_best_fit_residual_y_lim = ax_lc_global_best_fit_residual.get_ylim()
        else:
            ax_lc_global_best_fit.set_xlim(ax_lc_global_best_fit_x_lim)
            ax_lc_global_best_fit.set_ylim(ax_lc_global_best_fit_y_lim)
            ax_lc_global_best_fit_residual.set_xlim(ax_lc_global_best_fit_residual_x_lim)
            ax_lc_global_best_fit_residual.set_ylim(ax_lc_global_best_fit_residual_y_lim)

        lc_global_best_fit_plot.suptitle(f"{name} Sector {sector} {author} Global Best Fitted Light Curve And Residuals Exptime={exptime}s Interation {inter_global}")
        lc_global_best_fit_plot.figure.tight_layout()
        lc_global_best_fit_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01}-{inter_global} {name} Sector {sector} {author} Global Best Fitted Light Curve And Residuals Exptime={exptime}s Interation {inter_global}.png")


        # Remove the outliers
        lc_global_best_fit_residual_clipped, outliers_mask_global = lc_global_best_fit_residual.remove_outliers(sigma=sigma_global, return_mask=True)
        if np.sum(outliers_mask_global) == 0:
            break
        else:
            # update the lightcurve, initial parameters and transit model
            lc_corrected = lc_corrected[~outliers_mask_global]
            params_global_initial = params_global_best
            transit_model_global.set_data(lc_corrected.time.value)


    # Store the best fitted parameters in the dictionary
    params_global_best_dict['k'] = (k_global_best, k_global_best_lower_error, k_global_best_upper_error)
    params_global_best_dict['t0'] = (t0_global_best, t0_global_best_lower_error, t0_global_best_upper_error)
    params_global_best_dict['p'] = (p_global_best, p_global_best_lower_error, p_global_best_upper_error)
    params_global_best_dict['a'] = (a_global_best, a_global_best_lower_error, a_global_best_upper_error)
    params_global_best_dict['i'] = (i_global_best, i_global_best_lower_error, i_global_best_upper_error)
    params_global_best_dict['i_in_degree'] = (i_in_degree_global_best, i_in_degree_global_best_lower_error, i_in_degree_global_best_upper_error)
    params_global_best_dict['ldc1'] = (ldc1_global_best, ldc1_global_best_lower_error, ldc1_global_best_upper_error)
    params_global_best_dict['ldc2'] = (ldc2_global_best, ldc2_global_best_lower_error, ldc2_global_best_upper_error)
    params_global_best_dict['ldc'] = (ldc_global_best, ldc_global_best_lower_error, ldc_global_best_upper_error)

    params_global_best_dict['transit_duration'] = (transit_duration_global_best, transit_duration_global_best_lower_error, transit_duration_global_best_upper_error)
    params_global_best_dict['transit_duration_in_cadence'] = (transit_duration_in_cadence_global_best, transit_duration_in_cadence_global_best_lower_error, transit_duration_in_cadence_global_best_upper_error)
    params_global_best_dict['transit_depth'] = (transit_depth_global_best, transit_depth_global_best_lower_error, transit_depth_global_best_upper_error)

    params_global_best_dict['n_fitting_iteration'] = inter_global
    params_global_best_dict['residual_std'] = residual_std_global
    params_global_best_dict['chi_square'] = chi_square_global
    params_global_best_dict['reduced_chi_square'] = reduced_chi_square_global




### Individual Fitting ###
i += 1 # count the step


if fit_individual:
    # Initialize the parameters dictionary
    params_individual_best_dict = {
        'n_transit': None, 'n_valid_transit': None,
        'no_data_transit': [], 'valid_transit': [],
        'model': None,
        'k': [], 't0': [], 'p': None, 'a': [], 'i': [], 'i_in_degree': [], 'ldc1': [], 'ldc2': [], 'ldc': [],
        'transit_duration': [], 'transit_duration_in_cadence': [], 'transit_depth': [],
        'n_fitting_iteration': [], 'residual_std': [], 'chi_square': [], 'reduced_chi_square': []
    }

    params_individual_name = ['k', 't0', 'a', 'i', 'ldc1', 'ldc2']


    # Calculate the number of transits
    if fit_global:
        p = p_global_best
    else:
        p = p_raw_nans_removed
    n_transit = int(np.floor((lc_corrected.time.value[-1] - lc_corrected.time.value[0]) / p))

    # Split the light curve into individual parts, each containing one transit based on the global fitted period or the NaNs-removed raw light curve BLS-fitted period
    individual_mask_list = []
    lc_list_corrected_individual = []
    for transit_index in range(n_transit):
        individual_mask = ((lc_corrected.time.value >= lc_corrected.time.value[0] + transit_index * p) & (lc_corrected.time.value < lc_corrected.time.value[0] + (transit_index + 1) * p))
        lc_corrected_individual = lc_corrected[individual_mask]
        individual_mask_list.append(individual_mask)
        lc_list_corrected_individual.append(lc_corrected_individual)


    ##### Initialize the transit model #####
    transit_model_name_individual = 'Quadratic'
    transit_model_individual = QuadraticModel()


    ##### Set the initial individual parameters based on the best fitted global parameters or the NaNs-removed raw light curve BLS-fitted parameters #####
    if fit_global:
        k_individual_initial = k_global_best # k: normalized planetary radius, i.e., R_p/R_s
        p_individual = p # p: period, fixed
        a_individual_initial = a_global_best # a: normalized semi-major axis, i.e., a/R_s
        i_individual_initial = i_global_best # i: inclination in radians
        ldc1_individual_initial = ldc1_global_best # ldc1: linear coefficient
        ldc2_individual_initial = ldc2_global_best # ldc2: quadratic coefficient
        ldc_individual_initial = [ldc1_individual_initial, ldc2_individual_initial] # ldc: quadratic limb darkening coefficients
    else:
        k_individual_initial = np.sqrt(transit_depth_raw_nans_removed) # k: normalized planetary radius, i.e., R_p/R_s
        p_individual = p # p: period, fixed
        a_individual_initial = 10.0 # a: normalized semi-major axis, i.e., a/R_s
        i_individual_initial = np.pi / 2 # i: inclination in radians
        ldc1_individual_initial = 0.2 # ldc1: linear coefficient
        ldc2_individual_initial = 0.3 # ldc2: quadratic coefficient
        ldc_individual_initial = [ldc1_individual_initial, ldc2_individual_initial] # ldc: quadratic limb darkening coefficients


    # Store the parameters in the dictionary
    params_individual_best_dict['n_transit'] = n_transit
    params_individual_best_dict['model'] = transit_model_name_individual
    params_individual_best_dict['p'] = p_individual


    # Create the masked lightcurves lists for the all-in-one best fitted model light curves plot
    lc_list_corrected_individual_masked = []
    lc_list_individual_best_fit_masked = []
    lc_list_individual_best_fit_residual_masked = []
    individual_transit_min_model_flux_list = []


    for transit_index in range(n_transit):
        lc_corrected_individual = lc_list_corrected_individual[transit_index]

        # Check if the individual light curve has data points
        if lc_corrected_individual.time.size == 0:
            print(f"Warning: Transit {transit_index:02} has no data points, skipping.")
            params_individual_best_dict['no_data_transit'].append(transit_index)
            for key in ['k', 't0', 'a', 'i', 'i_in_degree', 'ldc1', 'ldc2', 'transit_duration', 'transit_duration_in_cadence', 'transit_depth']:
                params_individual_best_dict[key].append((np.nan, np.nan, np.nan))
            params_individual_best_dict['ldc'].append([(np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan)])
            params_individual_best_dict['n_fitting_iteration'].append(np.nan)
            params_individual_best_dict['residual_std'].append(np.nan)
            params_individual_best_dict['chi_square'].append(np.nan)
            params_individual_best_dict['reduced_chi_square'].append(np.nan)
            lc_list_corrected_individual_masked.append(None)
            lc_list_individual_best_fit_masked.append(None)
            lc_list_individual_best_fit_residual_masked.append(None)
            individual_transit_min_model_flux_list.append(np.nan)
            continue

        # Check if the individual light curve contains the transit event
        if fit_global:
            transit_start = t0_global_best + transit_index * p_individual - (transit_duration_global_best / 2) * individual_transit_check_coefficient
            transit_end = t0_global_best + transit_index * p_individual + (transit_duration_global_best / 2) * individual_transit_check_coefficient
        else:
            transit_start = t0_raw_nans_removed + transit_index * p_individual - (transit_duration_raw_nans_removed / 2) * transit_mask_raw_nans_removed_coefficient * individual_transit_check_coefficient
            tarnsit_end = t0_raw_nans_removed + transit_index * p_individual + (transit_duration_raw_nans_removed / 2) * transit_mask_raw_nans_removed_coefficient * individual_transit_check_coefficient

        transit_in_segment = np.any((lc_corrected_individual.time.value >= transit_start) & (lc_corrected_individual.time.value <= transit_end))
        if not transit_in_segment:
            print(f"Warning: Transit {transit_index:02} does not contain transit event, skipping.")
            params_individual_best_dict['no_data_transit'].append(transit_index)
            for key in ['k', 't0', 'a', 'i', 'i_in_degree', 'ldc1', 'ldc2', 'transit_duration', 'transit_duration_in_cadence', 'transit_depth']:
                params_individual_best_dict[key].append((np.nan, np.nan, np.nan))
            params_individual_best_dict['ldc'].append([(np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan)])
            params_individual_best_dict['n_fitting_iteration'].append(np.nan)
            params_individual_best_dict['residual_std'].append(np.nan)
            params_individual_best_dict['chi_square'].append(np.nan)
            params_individual_best_dict['reduced_chi_square'].append(np.nan)
            lc_list_corrected_individual_masked.append(None)
            lc_list_individual_best_fit_masked.append(None)
            lc_list_individual_best_fit_residual_masked.append(None)
            individual_transit_min_model_flux_list.append(np.nan)
            continue

        # Define the directories
        processed_lightcurve_plots_exptime_individual_transit_parent_dir = processed_lightcurve_plots_exptime_parent_dir + f"/Transit {transit_index:02}"
        os.makedirs(processed_lightcurve_plots_exptime_individual_transit_parent_dir, exist_ok=True)

        # Configure the transit model
        transit_model_individual.set_data(lc_corrected_individual.time.value)


        # Set the initial individual t0 specifically based on the best fitted global t0 or the NaNs-removed raw light curve BLS-fitted t0
        if fit_global:
            t0_individual_initial = t0_global_best + transit_index * p_individual # t0: epoch time
        else:
            t0_individual_initial = t0_raw_nans_removed + transit_index * p_individual # t0: epoch time
        params_individual_initial = [k_individual_initial, t0_individual_initial, a_individual_initial, i_individual_initial, ldc1_individual_initial, ldc2_individual_initial]


        inter_individual = 0
        while inter_individual < max_inter_individual:
            inter_individual += 1
            # Initialize walkers
            params_individual_position = params_individual_initial + 1e-4 * np.random.randn(n_walkers, n_dim_single)

            # Initialize and run the MCMC sampler
            params_individual_sampler = emcee.EnsembleSampler(n_walkers, n_dim_single, log_probability_single, args=(p_individual, transit_model_individual, lc_corrected_individual))
            params_individual_sampler.run_mcmc(params_individual_position, n_steps_individual, progress=True, progress_kwargs={'desc': f"Individual Fitting For Transit {transit_index:02} Interation {inter_individual}: "})

            # Retrieve the best fitted parameters and their uncertainties from the MCMC sampler
            params_individual_samples = params_individual_sampler.get_chain(discard=int(n_steps_individual * chain_discard_proportion_individual), flat=True)
            params_individual_best = np.median(params_individual_samples, axis=0)
            params_individual_best_lower_error = params_individual_best - np.percentile(params_individual_samples, 16, axis=0)
            params_individual_best_upper_error = np.percentile(params_individual_samples, 84, axis=0) - params_individual_best

            k_individual_samples, t0_individual_samples, a_individual_samples, i_individual_samples, ldc1_individual_samples, ldc2_individual_samples = params_individual_samples.T
            ldc_individual_samples = np.column_stack((ldc1_individual_samples, ldc2_individual_samples))

            k_individual_best, t0_individual_best, a_individual_best, i_individual_best, ldc1_individual_best, ldc2_individual_best = params_individual_best
            k_individual_best_lower_error, t0_individual_best_lower_error, a_individual_best_lower_error, i_individual_best_lower_error, ldc1_individual_best_lower_error, ldc2_individual_best_lower_error = params_individual_best_lower_error
            k_individual_best_upper_error, t0_individual_best_upper_error, a_individual_best_upper_error, i_individual_best_upper_error, ldc1_individual_best_upper_error, ldc2_individual_best_upper_error = params_individual_best_upper_error

            i_in_degree_individual_best, i_in_degree_individual_best_lower_error, i_in_degree_individual_best_upper_error = np.rad2deg(i_individual_best), np.rad2deg(i_individual_best_lower_error), np.rad2deg(i_individual_best_upper_error)
            ldc_individual_best, ldc_individual_best_lower_error, ldc_individual_best_upper_error = [ldc1_individual_best, ldc2_individual_best], [ldc1_individual_best_lower_error, ldc2_individual_best_lower_error], [ldc1_individual_best_upper_error, ldc2_individual_best_upper_error]

            # calculate the transit duration
            transit_duration_individual_samples = [calculate_transit_duration(k, p_individual, a, i) for k, a, i in zip(k_individual_samples, a_individual_samples, i_individual_samples)]
            transit_duration_individual_best = np.median(transit_duration_individual_samples, axis=0)
            transit_duration_individual_best_lower_error = transit_duration_individual_best - np.percentile(transit_duration_individual_samples, 16, axis=0)
            transit_duration_individual_best_upper_error = np.percentile(transit_duration_individual_samples, 84, axis=0) - transit_duration_individual_best

            # convert transit duration to number of cadences
            transit_duration_in_cadence_individual_best = int(round(transit_duration_individual_best / exptime_in_day))
            transit_duration_in_cadence_individual_best_lower_error = int(round(transit_duration_individual_best_lower_error / exptime_in_day))
            transit_duration_in_cadence_individual_best_upper_error = int(round(transit_duration_individual_best_upper_error / exptime_in_day))

            # calculate the normalized transit depth
            transit_depth_individual_samples = [calculate_transit_depth(k, a, i, ldc, transit_model_name_individual) for k, a, i, ldc in zip(k_individual_samples, a_individual_samples, i_individual_samples, ldc_individual_samples)]
            transit_depth_individual_best = np.median(transit_depth_individual_samples, axis=0)
            transit_depth_individual_best_lower_error = transit_depth_individual_best - np.percentile(transit_depth_individual_samples, 16, axis=0)
            transit_depth_individual_best_upper_error = np.percentile(transit_depth_individual_samples, 84, axis=0) - transit_depth_individual_best


            # calculate the best fitted model flux
            lc_individual_best_fit = lc_corrected_individual.copy()
            lc_individual_best_fit.flux = model_single(params_individual_best, p_individual, transit_model_individual, time=lc_individual_best_fit.time.value) * lc_corrected_individual.flux.unit
            lc_individual_best_fit.flux_err = np.zeros(len(lc_individual_best_fit.flux_err))

            # calculate the best fitted model residuals
            lc_individual_best_fit_residual = lc_corrected_individual.copy()
            lc_individual_best_fit_residual.flux = lc_corrected_individual.flux - lc_individual_best_fit.flux
            lc_individual_best_fit_residual.flux_err = lc_corrected_individual.flux_err - lc_individual_best_fit.flux_err

            # calculate the residual standard deviation
            residual_std_individual = np.std(lc_individual_best_fit_residual.flux.value)

            # calculate the chi-square and reduced chi-square of the best fitted model
            chi_square_individual = np.sum((lc_individual_best_fit_residual.flux.value / lc_corrected_individual.flux_err.value) ** 2)
            reduced_chi_square_individual = chi_square_individual / (len(lc_corrected_individual.flux.value) - n_dim_single)


            # Plot the visualization plots of the fitting process and result
            # Plot the parameters trace and evolution plots
            j = 1  # count the sub-step

            params_individual_samples_thinned_unflattened = params_individual_sampler.get_chain(discard=int(n_steps_individual * chain_discard_proportion_individual), thin=chain_thin_individual, flat=False)  # retrieve the thinned and unflattened sample chains from the MCMC sampler

            params_individual_trace_evolution_plot = plt.figure(figsize=(20, 2 * n_dim_single))
            params_individual_trace_evolution_gs = GridSpec(n_dim_single, 2, wspace=0.05)

            params_individual_moving_means = []
            for d in range(n_dim_single):
                param_individual_mean = np.mean(params_individual_samples_thinned_unflattened[:, :, d], axis=1)
                param_individual_moving_means = np.convolve(param_individual_mean, np.ones(running_mean_window_length_individual) / running_mean_window_length_individual, mode='valid')
                params_individual_moving_means.append(param_individual_moving_means)

            for d in range(n_dim_single):
                # Plot the parameter trace plot on the left side
                ax_trace_individual = params_individual_trace_evolution_plot.add_subplot(params_individual_trace_evolution_gs[d, 0])
                for w in range(n_walkers):
                    ax_trace_individual.plot(params_individual_samples_thinned_unflattened[:, w, d], alpha=0.5, linewidth=0.8)
                ax_trace_individual.set_ylabel(params_individual_name[d])
                ax_trace_individual.grid(True, alpha=0.3)
                # only show the x-axis label on the last subplot
                if d == n_dim_single - 1:
                    ax_trace_individual.set_xlabel("Step Number")
                else:
                    ax_trace_individual.tick_params(labelbottom=False)
                # only show the title on the first subplot
                if d == 0:
                    ax_trace_individual.set_title("Trace Of Parameters", fontsize='x-large')

                # Plot the parameter evolution plot on the right side
                ax_evolution_individual = params_individual_trace_evolution_plot.add_subplot(params_individual_trace_evolution_gs[d, 1], sharey=ax_trace_individual)
                ax_evolution_individual.plot(params_individual_moving_means[d], c='red')
                ax_evolution_individual.tick_params(labelleft=False)  # hide the y-axis labels
                ax_evolution_individual.grid(True, alpha=0.3)
                # only show the x-axis label on the last subplot
                if d == n_dim_single - 1:
                    ax_evolution_individual.set_xlabel("Step Number")
                else:
                    ax_evolution_individual.tick_params(labelbottom=False)
                # only show the legend on the first subplot
                if d == 0:
                    ax_evolution_individual.set_title(f"Evolution ({running_mean_window_proportion * 100}% Window Running Mean) Of Parameters", fontsize='x-large')

            params_individual_trace_evolution_plot.suptitle(f"{name} Sector {sector} {author} Individual Fitting Trace And Evolution Plot (Thinned By {chain_thin_individual}) Transit {transit_index:02} Exptime={exptime}s Interation {inter_individual}", fontsize='xx-large')
            params_individual_trace_evolution_plot.figure.subplots_adjust(wspace=0.05)
            params_individual_trace_evolution_plot.figure.savefig(processed_lightcurve_plots_exptime_individual_transit_parent_dir + f"/{i:02}-{j:01}-{transit_index:02}-{inter_individual} {name} Sector {sector} {author} Individual Fitting Trace And Evolution Plot Transit {transit_index:02} Exptime={exptime}s Interation {inter_individual}.png")


            # Plot the parameters posterior distribution plot
            j += 1  # count the sub-step

            params_individual_corner_plot = corner.corner(params_individual_samples, labels=params_individual_name, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".4f", figsize=(20, 25))
            params_individual_corner_plot.suptitle(f"{name} Sector {sector} {author} Individual Fitting Parameters Posterior Distribution Corner Plot Transit {transit_index:02} Exptime={exptime}s Interation {inter_individual}", fontsize='xx-large', y=1.05)
            params_individual_corner_plot.figure.savefig(processed_lightcurve_plots_exptime_individual_transit_parent_dir + f"/{i:02}-{j:01}-{transit_index:02}-{inter_individual} {name} Sector {sector} {author} Individual Fitting Parameters Posterior Distribution Corner Plot Transit {transit_index:02} Exptime={exptime}s Interation {inter_individual}.png", bbox_inches='tight')


            # Plot the best fitted model light curve and residuals
            j += 1  # count the sub-step

            # create the individual transit plot mask and apply it to the lightcurves
            individual_transit_plot_range = (t0_individual_best - transit_duration_individual_best / 2 * individual_transit_plot_coefficient, t0_individual_best + transit_duration_individual_best / 2 * individual_transit_plot_coefficient)
            individual_transit_plot_mask = ((lc_corrected_individual.time.value >= individual_transit_plot_range[0]) & (lc_corrected_individual.time.value < individual_transit_plot_range[1]))
            lc_corrected_individual_masked = lc_corrected_individual[individual_transit_plot_mask]
            lc_individual_best_fit_masked = lc_individual_best_fit[individual_transit_plot_mask]
            lc_individual_best_fit_residual_masked = lc_individual_best_fit_residual[individual_transit_plot_mask]
            individual_transit_min_model_flux = lc_individual_best_fit_masked.flux.value.min()

            lc_individual_best_fit_plot, (ax_lc_individual_best_fit, ax_lc_individual_best_fit_residual) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
            lc_corrected_individual_masked.scatter(ax=ax_lc_individual_best_fit, label=None, s=0.1, alpha=1.0)
            lc_corrected_individual_masked.errorbar(ax=ax_lc_individual_best_fit, label="Corrected Light Curve", alpha=alpha_exptime * 5 if alpha_exptime <= 1/5 else 1.0)
            lc_individual_best_fit_masked.plot(ax=ax_lc_individual_best_fit, c='red', label=f"Best Fitted {transit_model_name_individual} Model, chi-square={chi_square_individual:.2f}, reduced chi-square={reduced_chi_square_individual:.2f}")
            ax_lc_individual_best_fit.axhline(y=individual_transit_min_model_flux, c='black', linestyle='--', linewidth=1)

            # add the individual transit minimum model flux value to the y-axis
            ax_lc_individual_best_fit_y_ticks = ax_lc_individual_best_fit.get_yticks()
            ax_lc_individual_best_fit_y_ticks_interval = np.median(np.diff(ax_lc_individual_best_fit_y_ticks))
            ax_lc_individual_best_fit_y_ticks_closest_diff = np.min(np.abs(ax_lc_individual_best_fit_y_ticks - individual_transit_min_model_flux))
            ax_lc_individual_best_fit_y_ticks_closest_diff_idx = np.argmin(np.abs(ax_lc_individual_best_fit_y_ticks - individual_transit_min_model_flux))
            ax_lc_individual_best_fit_y_ticks_closest = ax_lc_individual_best_fit_y_ticks[ax_lc_individual_best_fit_y_ticks_closest_diff_idx]
            if ax_lc_individual_best_fit_y_ticks_closest_diff > ax_lc_individual_best_fit_y_ticks_interval * 0.2:
                ax_lc_individual_best_fit.text(x=0.0, y=individual_transit_min_model_flux, s=f"{individual_transit_min_model_flux:.6f}", transform=ax_lc_individual_best_fit.get_yaxis_transform(), ha='right', va='center')
            elif individual_transit_min_model_flux < ax_lc_individual_best_fit_y_ticks_closest:
                ax_lc_individual_best_fit.text(x=0.0, y=individual_transit_min_model_flux - ax_lc_individual_best_fit_y_ticks_interval * 0.2, s=f"{individual_transit_min_model_flux:.6f}", transform=ax_lc_individual_best_fit.get_yaxis_transform(), ha='right', va='center')
            elif individual_transit_min_model_flux > ax_lc_individual_best_fit_y_ticks_closest:
                ax_lc_individual_best_fit.text(x=0.0, y=individual_transit_min_model_flux + ax_lc_individual_best_fit_y_ticks_interval * 0.2, s=f"{individual_transit_min_model_flux:.6f}", transform=ax_lc_individual_best_fit.get_yaxis_transform(), ha='right', va='center')

            ax_lc_individual_best_fit.legend(loc='lower right')
            ax_lc_individual_best_fit.set_ylabel("Flux")
            ax_lc_individual_best_fit.set_xlabel("")
            # set the x-axis limits and y-axis limits of the latter interation to the same as the first one
            if inter_individual == 1:
                ax_lc_individual_best_fit.set_xlim(individual_transit_plot_range[0], individual_transit_plot_range[1])
                ax_lc_individual_best_fit_x_lim = ax_lc_individual_best_fit.get_xlim()
                ax_lc_individual_best_fit_y_lim = ax_lc_individual_best_fit.get_ylim()
            else:
                ax_lc_individual_best_fit.set_xlim(ax_lc_individual_best_fit_x_lim)
                ax_lc_individual_best_fit.set_ylim(ax_lc_individual_best_fit_y_lim) # plot the best fitted model flux

            lc_individual_best_fit_residual_masked.errorbar(ax=ax_lc_individual_best_fit_residual, c='green', label=f"Best Fitted Model Residuals, residual std={residual_std_individual:.6f}")
            ax_lc_individual_best_fit_residual.legend(loc='upper right')
            ax_lc_individual_best_fit_residual.set_ylabel("Residuals")
            # set the x-axis limits and y-axis limits of the latter interation to the same as the first one
            if inter_individual == 1:
                ax_lc_individual_best_fit_residual.set_xlim(individual_transit_plot_range[0], individual_transit_plot_range[1])
                ax_lc_individual_best_fit_residual_x_lim = ax_lc_individual_best_fit_residual.get_xlim()
                ax_lc_individual_best_fit_residual_y_lim = ax_lc_individual_best_fit_residual.get_ylim()
            else:
                ax_lc_individual_best_fit_residual.set_xlim(ax_lc_individual_best_fit_residual_x_lim)
                ax_lc_individual_best_fit_residual.set_ylim(ax_lc_individual_best_fit_residual_y_lim) # plot the best fitted model residuals

            lc_individual_best_fit_plot.suptitle(f"{name} Sector {sector} {author} Individual Best Fitted Light Curve And Residuals Transit {transit_index:02} Exptime={exptime}s Interation {inter_individual}")
            lc_individual_best_fit_plot.figure.tight_layout()
            lc_individual_best_fit_plot.figure.savefig(processed_lightcurve_plots_exptime_individual_transit_parent_dir + f"/{i:02}-{j:01}-{transit_index:02}-{inter_individual} {name} Sector {sector} {author} Individual Best Fitted Light Curve And Residuals Transit {transit_index:02} Exptime={exptime}s Interation {inter_individual}.png")


            # Remove the outliers
            lc_individual_best_fit_residual_clipped, outliers_mask_individual = lc_individual_best_fit_residual.remove_outliers(sigma=sigma_individual, return_mask=True)
            if np.sum(outliers_mask_individual) == 0:
                # store the masked lightcurves into a list when there are no outliers
                lc_list_corrected_individual_masked.append(lc_corrected_individual_masked)
                lc_list_individual_best_fit_masked.append(lc_individual_best_fit_masked)
                lc_list_individual_best_fit_residual_masked.append(lc_individual_best_fit_residual_masked)
                individual_transit_min_model_flux_list.append(individual_transit_min_model_flux)
                break
            else:
                # update the lightcurve, initial parameters and transit model
                lc_corrected_individual = lc_corrected_individual[~outliers_mask_individual]
                params_individual_initial = params_individual_best
                transit_model_individual.set_data(lc_corrected_individual.time.value)


        # Store the best fitted parameters in the dictionary
        params_individual_best_dict['k'].append([k_individual_best, k_individual_best_lower_error, k_individual_best_upper_error])
        params_individual_best_dict['t0'].append([t0_individual_best, t0_individual_best_lower_error, t0_individual_best_upper_error])
        params_individual_best_dict['a'].append([a_individual_best, a_individual_best_lower_error, a_individual_best_upper_error])
        params_individual_best_dict['i'].append([i_individual_best, i_individual_best_lower_error, i_individual_best_upper_error])
        params_individual_best_dict['i_in_degree'].append([i_in_degree_individual_best, i_in_degree_individual_best_lower_error, i_in_degree_individual_best_upper_error])
        params_individual_best_dict['ldc1'].append([ldc1_individual_best, ldc1_individual_best_lower_error, ldc1_individual_best_upper_error])
        params_individual_best_dict['ldc2'].append([ldc2_individual_best, ldc2_individual_best_lower_error, ldc2_individual_best_upper_error])
        params_individual_best_dict['ldc'].append([ldc_individual_best, ldc_individual_best_lower_error, ldc_individual_best_upper_error])

        params_individual_best_dict['transit_duration'].append([transit_duration_individual_best, transit_duration_individual_best_lower_error, transit_duration_individual_best_upper_error])
        params_individual_best_dict['transit_duration_in_cadence'].append([transit_duration_in_cadence_individual_best, transit_duration_in_cadence_individual_best_lower_error, transit_duration_in_cadence_individual_best_upper_error])
        params_individual_best_dict['transit_depth'].append([transit_depth_individual_best, transit_depth_individual_best_lower_error, transit_depth_individual_best_upper_error])

        params_individual_best_dict['n_fitting_iteration'].append(inter_individual)
        params_individual_best_dict['residual_std'].append(residual_std_individual)
        params_individual_best_dict['chi_square'].append(chi_square_individual)
        params_individual_best_dict['reduced_chi_square'].append(reduced_chi_square_individual)


    valid_transit = np.setdiff1d(np.arange(n_transit), params_individual_best_dict['no_data_transit'])
    n_valid_transit = len(valid_transit)
    # Store the parameters in the dictionary
    params_individual_best_dict['valid_transit'] = valid_transit
    params_individual_best_dict['n_valid_transit'] = n_valid_transit


    # Plot the all-in-one best fitted model light curves
    lc_individual_best_fit_all_plot, axes_lc_individual_best_fit = plt.subplots(n_valid_transit, 1, figsize=(20, 5 * n_valid_transit))
    if n_valid_transit == 1:
        axes_lc_individual_best_fit = [axes_lc_individual_best_fit]
    for plot_index, valid_transit_index in enumerate(valid_transit):
        individual_transit_plot_range = (params_individual_best_dict['t0'][valid_transit_index][0] - params_individual_best_dict['transit_duration'][valid_transit_index][0] / 2 * individual_transit_plot_coefficient,
                                         params_individual_best_dict['t0'][valid_transit_index][0] + params_individual_best_dict['transit_duration'][valid_transit_index][0] / 2 * individual_transit_plot_coefficient)
        lc_corrected_individual_masked = lc_list_corrected_individual_masked[valid_transit_index]
        lc_individual_best_fit_masked = lc_list_individual_best_fit_masked[valid_transit_index]
        individual_transit_min_model_flux = individual_transit_min_model_flux_list[valid_transit_index]

        ax_lc_individual_best_fit = axes_lc_individual_best_fit[plot_index]
        lc_corrected_individual_masked.scatter(ax=ax_lc_individual_best_fit, label=None, s=0.1, alpha=1.0)
        lc_corrected_individual_masked.errorbar(ax=ax_lc_individual_best_fit, label="Corrected Light Curve" if plot_index == 0 else None, alpha=alpha_exptime * 5 if alpha_exptime <= 1/5 else 1.0)
        lc_individual_best_fit_masked.plot(ax=ax_lc_individual_best_fit, c='red',
                                           label=f"Best Fitted {transit_model_name_individual} Model, chi-square={params_individual_best_dict['chi_square'][valid_transit_index]:.2f}, reduced chi-square={params_individual_best_dict['reduced_chi_square'][valid_transit_index]:.2f}, residual std={params_individual_best_dict['residual_std'][valid_transit_index]:.6f}"
                                           if plot_index == 0 else f"chi-square={params_individual_best_dict['chi_square'][valid_transit_index]:.2f}, reduced chi-square={params_individual_best_dict['reduced_chi_square'][valid_transit_index]:.2f}, residual std={params_individual_best_dict['residual_std'][valid_transit_index]:.6f}")
        ax_lc_individual_best_fit.axhline(y=individual_transit_min_model_flux, c='black', linestyle='--', linewidth=1)

        # add the individual transit minimum model flux value to the y-axis
        ax_lc_individual_best_fit_y_ticks = ax_lc_individual_best_fit.get_yticks()
        ax_lc_individual_best_fit_y_ticks_interval = np.median(np.diff(ax_lc_individual_best_fit_y_ticks))
        ax_lc_individual_best_fit_y_ticks_closest_diff = np.min(np.abs(ax_lc_individual_best_fit_y_ticks - individual_transit_min_model_flux))
        ax_lc_individual_best_fit_y_ticks_closest_diff_idx = np.argmin(np.abs(ax_lc_individual_best_fit_y_ticks - individual_transit_min_model_flux))
        ax_lc_individual_best_fit_y_ticks_closest = ax_lc_individual_best_fit_y_ticks[ax_lc_individual_best_fit_y_ticks_closest_diff_idx]
        if ax_lc_individual_best_fit_y_ticks_closest_diff > ax_lc_individual_best_fit_y_ticks_interval * 0.3:
            ax_lc_individual_best_fit.text(x=0.0, y=individual_transit_min_model_flux, s=f"{individual_transit_min_model_flux:.6f}", transform=ax_lc_individual_best_fit.get_yaxis_transform(), ha='right', va='center')
        elif individual_transit_min_model_flux < ax_lc_individual_best_fit_y_ticks_closest:
            ax_lc_individual_best_fit.text(x=0.0, y=individual_transit_min_model_flux - ax_lc_individual_best_fit_y_ticks_interval * 0.3, s=f"{individual_transit_min_model_flux:.6f}", transform=ax_lc_individual_best_fit.get_yaxis_transform(), ha='right', va='center')
        elif individual_transit_min_model_flux > ax_lc_individual_best_fit_y_ticks_closest:
            ax_lc_individual_best_fit.text(x=0.0, y=individual_transit_min_model_flux + ax_lc_individual_best_fit_y_ticks_interval * 0.3, s=f"{individual_transit_min_model_flux:.6f}", transform=ax_lc_individual_best_fit.get_yaxis_transform(), ha='right', va='center')

        ax_lc_individual_best_fit.legend(loc='lower right')
        ax_lc_individual_best_fit.set_ylabel("Flux")
        ax_lc_individual_best_fit.set_xlim(individual_transit_plot_range[0], individual_transit_plot_range[1])
        ax_lc_individual_best_fit.set_title(f"Transit {valid_transit_index:02}", fontsize='x-large')
    lc_individual_best_fit_all_plot.suptitle(f"{name} Sector {sector} {author} Individual Best Fitted Light Curves Exptime={exptime}s", fontsize='xx-large', y=1.02)
    lc_individual_best_fit_all_plot.tight_layout()
    lc_individual_best_fit_all_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02} {name} Sector {sector} {author} Individual Best Fitted Light Curves Exptime={exptime}s.png", bbox_inches='tight')




##### Define the Lightkurve parameters depending on the exptime and fitted parameters #####
bin = True
time_bin_size = exptime * u.second


##### Define the emcee MCMC sampling parameters #####
fit_binned = True

max_inter_binned = 3 # maximum number of iterations for the individual fitting after removing outliers
sigma_binned = 3.0 # sigma for the individual fitting outlier removal

n_steps_binned = 5000
chain_discard_proportion_binned = 0.2 # the proportion of the chain to discard to remove the burn-in phase

chain_thin_binned = 10 # thinning factor of the sample chain when visualizing the process and result
running_mean_window_length_binned = int(running_mean_window_proportion * n_steps_binned * (1 - chain_discard_proportion_binned) / chain_thin_binned)
binned_transit_plot_coefficient = 2 # the coefficient of the binned transit plot span




### Fold ###
# Fold the lightcurve based on the fitted parameters and plot the folded light curve
i += 1 # count the step


if fit_global:
    p_folded = p_global_best
    t0_folded = t0_global_best
else:
    p_folded = p_raw_nans_removed
    t0_folded = t0_raw_nans_removed
lc_folded = lc_corrected.fold(period=p_folded, epoch_time=t0_folded)
lc_folded_cdpp = calculate_cdpp(lc_folded, transit_duration=transit_duration_in_cadence_raw_nans_removed)

lc_folded_plot, ax_lc_folded = plt.subplots(figsize=(10, 5))
lc_folded.scatter(ax=ax_lc_folded, label=None, s=0.1, alpha=1.0)
lc_folded.errorbar(ax=ax_lc_folded, label=f"simplified CDPP={lc_folded_cdpp:.2f}", alpha=alpha_exptime / 2)
ax_lc_folded.legend(loc='lower right')
ax_lc_folded.set_title(f"{name} Sector {sector} {author} Period={p_folded:.4f}d Folded Light Curve Exptime={exptime}s")
lc_folded_plot.figure.tight_layout()
lc_folded_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02} {name} Sector {sector} {author} Period={p_folded}d Folded Light Curve Exptime={exptime}s.png")




### Bin ###
# Bin the lightcurve and plot the binned light curve
i += 1 # count the step


if bin:
    lc_binned = lc_folded.bin(time_bin_size=time_bin_size)
    lc_binned_cdpp = calculate_cdpp(lc_binned, transit_duration=transit_duration_in_cadence_raw_nans_removed)

    lc_binned_plot, ax_lc_binned = plt.subplots(figsize=(10, 5))
    lc_binned.scatter(ax=ax_lc_binned, label=None, s=0.1, alpha=1.0)
    lc_binned.errorbar(ax=ax_lc_binned, label=f"simplified CDPP={lc_binned_cdpp:.2f}", alpha=alpha_exptime * 5 if alpha_exptime <= 1/5 else 1.0)
    ax_lc_binned.legend(loc='lower right')
    ax_lc_binned.set_title(f"{name} Sector {sector} {author} Time_Bin_Size={time_bin_size:.1f} Binned Light Curve Exptime={exptime}s")
    lc_binned_plot.figure.tight_layout()
    lc_binned_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02} {name} Sector {sector} {author} Time_Bin_Size={time_bin_size.value}s Binned Light Curve Exptime={exptime}s.png")


else:
    lc_binned = lc_folded.copy()
    lc_binned_cdpp = calculate_cdpp(lc_binned, transit_duration=transit_duration_in_cadence_raw_nans_removed)




### Binned Fitting ###
i += 1 # count the step


if fit_binned:
    # Initialize the parameters dictionary
    params_binned_best_dict = {
        'model': None,
        'k': None, 't0': None, 'p': None, 'a': None, 'i': None, 'i_in_degree': None, 'ldc1': None, 'ldc2': None, 'ldc': None,
        'transit_duration': None, 'transit_duration_in_cadence': None, 'transit_depth': None,
        'n_fitting_iteration': None, 'residual_std': None, 'chi_square': None, 'reduced_chi_square': None
    }

    params_binned_name = ['k', 't0', 'a', 'i', 'ldc1', 'ldc2']


    ##### Initialize and configure the transit model #####
    transit_model_name_binned = 'Quadratic'
    transit_model_binned = QuadraticModel()
    transit_model_binned.set_data(lc_binned.time.value)


    ##### Set the initial binned parameters based on the best fitted global parameters or the NaNs-removed raw light curve BLS-fitted parameters #####
    if fit_global:
        k_binned_initial = k_global_best # k: normalized planetary radius, i.e., R_p/R_s
        t0_binned_initial = 0.0 # t0: epoch time
        # t0_binned_initial = t0_global_best - lc_corrected.time.value[0] # t0: epoch time
        p_binned = p_global_best # p: period, fixed
        a_binned_initial = a_global_best # a: normalized semi-major axis, i.e., a/R_s
        i_binned_initial = i_global_best # i: inclination in radians
        ldc1_binned_initial = ldc1_global_best # ldc1: linear coefficient
        ldc2_binned_initial = ldc2_global_best # ldc2: quadratic coefficient
        ldc_binned_initial = [ldc1_binned_initial, ldc2_binned_initial] # ldc: quadratic limb darkening coefficients
    else:
        k_binned_initial = np.sqrt(transit_depth_raw_nans_removed) # k: normalized planetary radius, i.e., R_p/R_s
        t0_binned_initial = 0.0 # t0: epoch time
        # t0_binned_initial = t0_raw_nans_removed - lc_corrected.time.value[0] # t0: epoch time
        p_binned = p_raw_nans_removed # p: period, fixed
        a_binned_initial = 10.0 # a: normalized semi-major axis, i.e., a/R_s
        i_binned_initial = np.pi / 2 # i: inclination in radians
        ldc1_binned_initial = 0.2 # ldc1: linear coefficient
        ldc2_binned_initial = 0.3 # ldc2: quadratic coefficient
        ldc_binned_initial = [ldc1_binned_initial, ldc2_binned_initial] # ldc: quadratic limb darkening coefficients

    params_binned_initial = [k_binned_initial, t0_binned_initial, a_binned_initial, i_binned_initial, ldc1_binned_initial, ldc2_binned_initial]


    # Store the parameters in the dictionary
    params_binned_best_dict['model'] = transit_model_name_binned
    params_binned_best_dict['p'] = p_binned


    inter_binned = 0
    while inter_binned < max_inter_binned:
        inter_binned += 1
        # Initialize walkers
        params_binned_position = params_binned_initial + 1e-4 * np.random.randn(n_walkers, n_dim_single)

        # Initialize and run the MCMC sampler
        params_binned_sampler = emcee.EnsembleSampler(n_walkers, n_dim_single, log_probability_single, args=(p_binned, transit_model_binned, lc_binned))
        params_binned_sampler.run_mcmc(params_binned_position, n_steps_binned, progress=True, progress_kwargs={'desc': f"Binned Fitting Interation {inter_binned}: "})

        # Retrieve the best fitted parameters and their uncertainties from the MCMC sampler
        params_binned_samples = params_binned_sampler.get_chain(discard=int(n_steps_binned * chain_discard_proportion_binned), flat=True)
        params_binned_best = np.median(params_binned_samples, axis=0)
        params_binned_best_lower_error = params_binned_best - np.percentile(params_binned_samples, 16, axis=0)
        params_binned_best_upper_error = np.percentile(params_binned_samples, 84, axis=0) - params_binned_best

        k_binned_samples, t0_binned_samples, a_binned_samples, i_binned_samples, ldc1_binned_samples, ldc2_binned_samples = params_binned_samples.T
        ldc_binned_samples = np.column_stack((ldc1_binned_samples, ldc2_binned_samples))

        k_binned_best, t0_binned_best, a_binned_best, i_binned_best, ldc1_binned_best, ldc2_binned_best = params_binned_best
        k_binned_best_lower_error, t0_binned_best_lower_error, a_binned_best_lower_error, i_binned_best_lower_error, ldc1_binned_best_lower_error, ldc2_binned_best_lower_error = params_binned_best_lower_error
        k_binned_best_upper_error, t0_binned_best_upper_error, a_binned_best_upper_error, i_binned_best_upper_error, ldc1_binned_best_upper_error, ldc2_binned_best_upper_error = params_binned_best_upper_error

        i_in_degree_binned_best, i_in_degree_binned_best_lower_error, i_in_degree_binned_best_upper_error = np.rad2deg(i_binned_best), np.rad2deg(i_binned_best_lower_error), np.rad2deg(i_binned_best_upper_error)
        ldc_binned_best, ldc_binned_best_lower_error, ldc_binned_best_upper_error = [ldc1_binned_best, ldc2_binned_best], [ldc1_binned_best_lower_error, ldc2_binned_best_lower_error], [ldc1_binned_best_upper_error, ldc2_binned_best_upper_error]

        # calculate the transit duration
        transit_duration_binned_samples = [calculate_transit_duration(k, p_binned, a, i) for k, a, i in zip(k_binned_samples, a_binned_samples, i_binned_samples)]
        transit_duration_binned_best = np.median(transit_duration_binned_samples, axis=0)
        transit_duration_binned_best_lower_error = transit_duration_binned_best - np.percentile(transit_duration_binned_samples, 16, axis=0)
        transit_duration_binned_best_upper_error = np.percentile(transit_duration_binned_samples, 84, axis=0) - transit_duration_binned_best

        # convert transit duration to number of cadences
        transit_duration_in_cadence_binned_best = int(round(transit_duration_binned_best / exptime_in_day))
        transit_duration_in_cadence_binned_best_lower_error = int(round(transit_duration_binned_best_lower_error / exptime_in_day))
        transit_duration_in_cadence_binned_best_upper_error = int(round(transit_duration_binned_best_upper_error / exptime_in_day))

        # calculate the normalized transit depth
        transit_depth_binned_samples = [calculate_transit_depth(k, a, i, ldc, transit_model_name_binned) for k, a, i, ldc in zip(k_binned_samples, a_binned_samples, i_binned_samples, ldc_binned_samples)]
        transit_depth_binned_best = np.median(transit_depth_binned_samples, axis=0)
        transit_depth_binned_best_lower_error = transit_depth_binned_best - np.percentile(transit_depth_binned_samples, 16, axis=0)
        transit_depth_binned_best_upper_error = np.percentile(transit_depth_binned_samples, 84, axis=0) - transit_depth_binned_best


        # calculate the best fitted model flux
        lc_binned_best_fit = lc_binned.copy()
        lc_binned_best_fit.flux = model_single(params_binned_best, p_binned, transit_model_binned, time=lc_binned_best_fit.time.value) * lc_binned.flux.unit
        lc_binned_best_fit.flux_err = np.zeros(len(lc_binned_best_fit.flux_err))

        # calculate the best fitted model residuals
        lc_binned_best_fit_residual = lc_binned.copy()
        lc_binned_best_fit_residual.flux = lc_binned.flux - lc_binned_best_fit.flux
        lc_binned_best_fit_residual.flux_err = lc_binned.flux_err - lc_binned_best_fit.flux_err

        # calculate the residual standard deviation
        residual_std_binned = np.std(lc_binned_best_fit_residual.flux.value)

        # calculate the chi-square and reduced chi-square of the best fitted model
        chi_square_binned = np.sum((lc_binned_best_fit_residual.flux.value / lc_binned.flux_err.value) ** 2)
        reduced_chi_square_binned = chi_square_binned / (len(lc_binned.flux.value) - n_dim_single)


        # Plot the visualization plots of the fitting process and result
        # Plot the parameters trace and evolution plots
        j = 1  # count the sub-step

        params_binned_samples_thinned_unflattened = params_binned_sampler.get_chain(discard=int(n_steps_binned * chain_discard_proportion_binned), thin=chain_thin_binned, flat=False)  # retrieve the thinned and unflattened sample chains from the MCMC sampler

        params_binned_trace_evolution_plot = plt.figure(figsize=(20, 2 * n_dim_single))
        params_binned_trace_evolution_gs = GridSpec(n_dim_single, 2, wspace=0.05)

        params_binned_moving_means = []
        for d in range(n_dim_single):
            param_binned_mean = np.mean(params_binned_samples_thinned_unflattened[:, :, d], axis=1)
            param_binned_moving_means = np.convolve(param_binned_mean, np.ones(running_mean_window_length_binned) / running_mean_window_length_binned, mode='valid')
            params_binned_moving_means.append(param_binned_moving_means)

        for d in range(n_dim_single):
            # Plot the parameter trace plot on the left side
            ax_trace_binned = params_binned_trace_evolution_plot.add_subplot(params_binned_trace_evolution_gs[d, 0])
            for w in range(n_walkers):
                ax_trace_binned.plot(params_binned_samples_thinned_unflattened[:, w, d], alpha=0.5, linewidth=0.8)
            ax_trace_binned.set_ylabel(params_binned_name[d])
            ax_trace_binned.grid(True, alpha=0.3)
            # only show the x-axis label on the last subplot
            if d == n_dim_single - 1:
                ax_trace_binned.set_xlabel("Step Number")
            else:
                ax_trace_binned.tick_params(labelbottom=False)
            # only show the title on the first subplot
            if d == 0:
                ax_trace_binned.set_title("Trace Of Parameters", fontsize='x-large')

            # Plot the parameter evolution plot on the right side
            ax_evolution_binned = params_binned_trace_evolution_plot.add_subplot(params_binned_trace_evolution_gs[d, 1], sharey=ax_trace_binned)
            ax_evolution_binned.plot(params_binned_moving_means[d], c='red')
            ax_evolution_binned.tick_params(labelleft=False)  # hide the y-axis labels
            ax_evolution_binned.grid(True, alpha=0.3)
            # only show the x-axis label on the last subplot
            if d == n_dim_single - 1:
                ax_evolution_binned.set_xlabel("Step Number")
            else:
                ax_evolution_binned.tick_params(labelbottom=False)
            # only show the legend on the first subplot
            if d == 0:
                ax_evolution_binned.set_title(f"Evolution ({running_mean_window_proportion * 100}% Window Running Mean) Of Parameters", fontsize='x-large')

        params_binned_trace_evolution_plot.suptitle(f"{name} Sector {sector} {author} Binned Fitting Trace And Evolution Plot (Thinned By {chain_thin_binned}) Exptime={exptime}s Interation {inter_binned}", fontsize='xx-large')
        params_binned_trace_evolution_plot.figure.subplots_adjust(wspace=0.05)
        params_binned_trace_evolution_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01}-{inter_binned} {name} Sector {sector} {author} Binned Fitting Trace And Evolution Plot Exptime={exptime}s Interation {inter_binned}.png")


        # Plot the parameters posterior distribution plot
        j += 1  # count the sub-step

        params_binned_corner_plot = corner.corner(params_binned_samples, labels=params_binned_name, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".4f", figsize=(20, 25))
        params_binned_corner_plot.suptitle(f"{name} Sector {sector} {author} Binned Fitting Parameters Posterior Distribution Corner Plot Exptime={exptime}s Interation {inter_binned}", fontsize='xx-large', y=1.05)
        params_binned_corner_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01}-{inter_binned} {name} Sector {sector} {author} Binned Fitting Parameters Posterior Distribution Corner Plot Exptime={exptime}s Interation {inter_binned}.png", bbox_inches='tight')


        # Plot the best fitted model light curve and residuals
        j += 1  # count the sub-step

        # create the binned transit plot mask and apply it to the lightcurves
        binned_transit_plot_range = (t0_binned_best - transit_duration_binned_best / 2 * binned_transit_plot_coefficient, t0_binned_best + transit_duration_binned_best / 2 * binned_transit_plot_coefficient)
        binned_transit_plot_mask = ((lc_binned.time.value >= binned_transit_plot_range[0]) & (lc_binned.time.value < binned_transit_plot_range[1]))
        lc_binned_masked = lc_binned[binned_transit_plot_mask]
        lc_binned_best_fit_masked = lc_binned_best_fit[binned_transit_plot_mask]
        lc_binned_best_fit_residual_masked = lc_binned_best_fit_residual[binned_transit_plot_mask]
        binned_transit_min_model_flux = lc_binned_best_fit_masked.flux.value.min()

        lc_binned_best_fit_plot, (ax_lc_binned_best_fit, ax_lc_binned_best_fit_residual) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
        lc_binned_masked.scatter(ax=ax_lc_binned_best_fit, label=None, s=0.1, alpha=1.0)
        lc_binned_masked.errorbar(ax=ax_lc_binned_best_fit, label="Binned Corrected Light Curve", alpha=alpha_exptime * 5 if alpha_exptime <= 1/5 else 1.0)
        lc_binned_best_fit_masked.plot(ax=ax_lc_binned_best_fit, c='red', label=f"Best Fitted {transit_model_name_binned} Model, chi-square={chi_square_binned:.2f}, reduced chi-square={reduced_chi_square_binned:.2f}")
        ax_lc_binned_best_fit.axhline(y=binned_transit_min_model_flux, c='black', linestyle='--', linewidth=1)

        # add the binned transit minimum model flux value to the y-axis
        ax_lc_binned_best_fit_y_ticks = ax_lc_binned_best_fit.get_yticks()
        ax_lc_binned_best_fit_y_ticks_interval = np.median(np.diff(ax_lc_binned_best_fit_y_ticks))
        ax_lc_binned_best_fit_y_ticks_closest_diff = np.min(np.abs(ax_lc_binned_best_fit_y_ticks - binned_transit_min_model_flux))
        ax_lc_binned_best_fit_y_ticks_closest_diff_idx = np.argmin(np.abs(ax_lc_binned_best_fit_y_ticks - binned_transit_min_model_flux))
        ax_lc_binned_best_fit_y_ticks_closest = ax_lc_binned_best_fit_y_ticks[ax_lc_binned_best_fit_y_ticks_closest_diff_idx]
        if ax_lc_binned_best_fit_y_ticks_closest_diff > ax_lc_binned_best_fit_y_ticks_interval * 0.2:
            ax_lc_binned_best_fit.text(x=0.0, y=binned_transit_min_model_flux, s=f"{binned_transit_min_model_flux:.6f}", transform=ax_lc_binned_best_fit.get_yaxis_transform(), ha='right', va='center')
        elif binned_transit_min_model_flux < ax_lc_binned_best_fit_y_ticks_closest:
            ax_lc_binned_best_fit.text(x=0.0, y=binned_transit_min_model_flux - ax_lc_binned_best_fit_y_ticks_interval * 0.2, s=f"{binned_transit_min_model_flux:.6f}", transform=ax_lc_binned_best_fit.get_yaxis_transform(), ha='right', va='center')
        elif binned_transit_min_model_flux > ax_lc_binned_best_fit_y_ticks_closest:
            ax_lc_binned_best_fit.text(x=0.0, y=binned_transit_min_model_flux + ax_lc_binned_best_fit_y_ticks_interval * 0.2, s=f"{binned_transit_min_model_flux:.6f}", transform=ax_lc_binned_best_fit.get_yaxis_transform(), ha='right', va='center')

        ax_lc_binned_best_fit.legend(loc='lower right')
        ax_lc_binned_best_fit.set_ylabel("Flux")
        ax_lc_binned_best_fit.set_xlabel("")
        # set the x-axis limits and y-axis limits of the latter interation to the same as the first one
        if inter_binned == 1:
            ax_lc_binned_best_fit.set_xlim(binned_transit_plot_range[0], binned_transit_plot_range[1])
            ax_lc_binned_best_fit_x_lim = ax_lc_binned_best_fit.get_xlim()
            ax_lc_binned_best_fit_y_lim = ax_lc_binned_best_fit.get_ylim()
        else:
            ax_lc_binned_best_fit.set_xlim(ax_lc_binned_best_fit_x_lim)
            ax_lc_binned_best_fit.set_ylim(ax_lc_binned_best_fit_y_lim) # plot the best fitted model flux


        lc_binned_best_fit_residual_masked.errorbar(ax=ax_lc_binned_best_fit_residual, c='green', label=f"Best Fitted Model Residuals, residual std={residual_std_binned:.6f}")
        ax_lc_binned_best_fit_residual.legend(loc='upper right')
        ax_lc_binned_best_fit_residual.set_ylabel("Residuals")
        # set the x-axis limits and y-axis limits of the latter interation to the same as the first one
        if inter_binned == 1:
            ax_lc_binned_best_fit_residual.set_xlim(binned_transit_plot_range[0], binned_transit_plot_range[1])
            ax_lc_binned_best_fit_residual_x_lim = ax_lc_binned_best_fit_residual.get_xlim()
            ax_lc_binned_best_fit_residual_y_lim = ax_lc_binned_best_fit_residual.get_ylim()
        else:
            ax_lc_binned_best_fit_residual.set_xlim(ax_lc_binned_best_fit_residual_x_lim)
            ax_lc_binned_best_fit_residual.set_ylim(ax_lc_binned_best_fit_residual_y_lim) # plot the best fitted model residuals

        lc_binned_best_fit_plot.suptitle(f"{name} Sector {sector} {author} Binned Best Fitted Light Curve And Residuals Exptime={exptime}s Interation {inter_binned}")
        lc_binned_best_fit_plot.figure.tight_layout()
        lc_binned_best_fit_plot.figure.savefig(processed_lightcurve_plots_exptime_parent_dir + f"/{i:02}-{j:01}-{inter_binned} {name} Sector {sector} {author} Binned Best Fitted Light Curve And Residuals Exptime={exptime}s Interation {inter_binned}.png")


        # Remove the outliers
        lc_binned_best_fit_residual_clipped, outliers_mask_binned = lc_binned_best_fit_residual.remove_outliers(sigma=sigma_binned, return_mask=True)
        if np.sum(outliers_mask_binned) == 0:
            break
        else:
            # update the lightcurve, initial parameters and transit model
            lc_binned = lc_binned[~outliers_mask_binned]
            params_binned_initial = params_binned_best
            transit_model_binned.set_data(lc_binned.time.value)


    # Store the best fitted parameters in the dictionary
    params_binned_best_dict['k'] = (k_binned_best, k_binned_best_lower_error, k_binned_best_upper_error)
    params_binned_best_dict['t0'] = (t0_binned_best, t0_binned_best_lower_error, t0_binned_best_upper_error)
    params_binned_best_dict['a'] = (a_binned_best, a_binned_best_lower_error, a_binned_best_upper_error)
    params_binned_best_dict['i'] = (i_binned_best, i_binned_best_lower_error, i_binned_best_upper_error)
    params_binned_best_dict['i_in_degree'] = (i_in_degree_binned_best, i_in_degree_binned_best_lower_error, i_in_degree_binned_best_upper_error)
    params_binned_best_dict['ldc1'] = (ldc1_binned_best, ldc1_binned_best_lower_error, ldc1_binned_best_upper_error)
    params_binned_best_dict['ldc2'] = (ldc2_binned_best, ldc2_binned_best_lower_error, ldc2_binned_best_upper_error)
    params_binned_best_dict['ldc'] = (ldc_binned_best, ldc_binned_best_lower_error, ldc_binned_best_upper_error)

    params_binned_best_dict['transit_duration'] = (transit_duration_binned_best, transit_duration_binned_best_lower_error, transit_duration_binned_best_upper_error)
    params_binned_best_dict['transit_duration_in_cadence'] = (transit_duration_in_cadence_binned_best, transit_duration_in_cadence_binned_best_lower_error, transit_duration_in_cadence_binned_best_upper_error)
    params_binned_best_dict['transit_depth'] = (transit_depth_binned_best, transit_depth_binned_best_lower_error, transit_depth_binned_best_upper_error)

    params_binned_best_dict['n_fitting_iteration'] = inter_binned
    params_binned_best_dict['residual_std'] = residual_std_binned
    params_binned_best_dict['chi_square'] = chi_square_binned
    params_binned_best_dict['reduced_chi_square'] = reduced_chi_square_binned




### ------ Documentation ------ ###
# Define the format_with_uncertainty() function to format the parameters with uncertainties
def format_with_uncertainty(param, precision=8):
    if isinstance(param, (tuple, list)) and len(param) == 3:
        val, lower, upper = param
        return f"{val:.{precision}f} -{abs(lower):.{precision}f}/+{abs(upper):.{precision}f}"
    else:
        return f"{param:.{precision}f}"




# Print the methodologies and results
methodology_result_file = open(processed_lightcurve_plots_exptime_parent_dir + f"/{name} Sector {sector} {author} {method} Exptime={exptime}s Methodologies And Results.txt", "w", encoding='utf-8')
methodology_result_file.write(f"{name} Sector {sector} {author} {method} Exptime={exptime}s Methodologies And Results\n\n")


methodology_result_file.write("NASA Exoplanet Archive Source Planetary Parameters: \n"
                                    f"Period (Days): {period_NASA}\n"
                                    f"Epoch Time (BTJD): {epoch_time_NASA}\n"
                                    f"Transit Duration (Days): {transit_duration_NASA}\n"
                                    f"Transit Depth: {transit_depth_NASA}\n\n")


methodology_result_file.write("-----------------------------Lightkurve Processing-----------------------------\n\n")


methodology_result_file.write(      f"Raw Light Curve CDPP: {lc_raw_cdpp}\n"
                                    f"NaNs-Removed Raw Baseline CDPP: {lc_raw_nans_removed_baseline_cdpp}\n"
                                    f"NaNs-Removed Raw Light Curve CDPP: {lc_raw_nans_removed_cdpp}\n\n")


methodology_result_file.write(      f"Box Least Squares (BLS) Fitted NaNs-Removed Raw Light Curve Parameters: \n"
                                    f"Period (Days): {p_raw_nans_removed}\n"
                                    f"Epoch Time (BTJD): {t0_raw_nans_removed}\n"
                                    f"Transit Duration (Days): {transit_duration_raw_nans_removed}\n"
                                    f"Transit Duration (Cadences): {transit_duration_in_cadence_raw_nans_removed}\n"
                                    f"Transit Depth: {transit_depth_raw_nans_removed}\n\n")


methodology_result_file.write(      f"Flatten: {flatten}\n")
if flatten:
    methodology_result_file.write(f"Flatten Window Proportion: {flatten_window_proportion}\n"
                                  f"Flatten Window Length: {flatten_window_length}\n"
                                  f"Flatten Polyorder: {flatten_polyorder}\n"
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


methodology_result_file.write(      f"Correction: {correction}\n\n")


if fit_global:
    methodology_result_file.write("------------------------------------Global Fitting------------------------------------\n\n")


    methodology_result_file.write(      f"MCMC Steps: {n_steps_global}\n"
                                        f"Chain Discard Proportion: {chain_discard_proportion_global}\n\n")


    methodology_result_file.write(      f"{params_global_best_dict['model']} Model Fitted Corrected Light Curve Global Parameters: \n"
                                        f"Normalized Planetary Radius (k, R_p/R_s): {format_with_uncertainty(params_global_best_dict['k'])}\n"
                                        f"Epoch Time (t0): {format_with_uncertainty(params_global_best_dict['t0'])}\n"
                                        f"Period (p): {format_with_uncertainty(params_global_best_dict['p'])}\n"
                                        f"Normalized Semi-Major Axis (a/R_s): {format_with_uncertainty(params_global_best_dict['a'])}\n"
                                        f"Inclination (i) (Radians): {format_with_uncertainty(params_global_best_dict['i'])}\n"
                                        f"Inclination (i) (Degrees): {format_with_uncertainty(params_global_best_dict['i_in_degree'])}\n"
                                        f"Quadratic Limb Darkening Coefficients (ldc1, ldc2): {format_with_uncertainty(params_global_best_dict['ldc1'])}, {format_with_uncertainty(params_global_best_dict['ldc2'])}\n"
                                        f"Transit Duration (Days): {format_with_uncertainty(params_global_best_dict['transit_duration'])}\n"
                                        f"Transit Duration (Cadences): {format_with_uncertainty(params_global_best_dict['transit_duration_in_cadence'], precision=0)}\n"
                                        f"Transit Depth: {format_with_uncertainty(params_global_best_dict['transit_depth'])}\n"
                                        f"Fitting Interation: {params_global_best_dict['n_fitting_iteration']}\n"
                                        f"Residual Standard Deviation: {params_global_best_dict['residual_std']:.6f}\n"
                                        f"Chi-Square: {params_global_best_dict['chi_square']:.2f}\n"
                                        f"Reduced Chi-Square: {params_global_best_dict['reduced_chi_square']:.2f}\n\n")
else:
    methodology_result_file.write("\n") # write an empty line


if fit_individual:
    methodology_result_file.write("--------------------------------Individual Fitting--------------------------------\n\n")


    methodology_result_file.write(      f"MCMC Steps: {n_steps_individual}\n"
                                        f"Chain Discard Proportion: {chain_discard_proportion_individual}\n\n")


    methodology_result_file.write(      f"{params_individual_best_dict['model']} Model Fitted Corrected Light Curve Individual Parameters: \n"
                                        f"Number Of Transits: {params_individual_best_dict['n_transit']}\n"
                                        f"Number Of Valid Transits: {params_individual_best_dict['n_valid_transit']}\n"
                                        f"Transit(s) With No Data Points: {', '.join([f'{no_data_transit}' for no_data_transit in params_individual_best_dict['no_data_transit']])}\n"
                                        f"Valid Transit(s): {', '.join([f'{valid_transit}' for valid_transit in params_individual_best_dict['valid_transit']])}\n"
                                        f"Normalized Planetary Radius (k, R_p/R_s): {', '.join([format_with_uncertainty(k) for k in params_individual_best_dict['k']])}\n"
                                        f"Epoch Time (t0): {', '.join([format_with_uncertainty(t0) for t0 in params_individual_best_dict['t0']])}\n"
                                        f"Period (p) (fixed to the same as the fitted period): {params_individual_best_dict['p']}\n"
                                        f"Normalized Semi-Major Axis (a/R_s): {', '.join([format_with_uncertainty(a) for a in params_individual_best_dict['a']])}\n"
                                        f"Inclination (i) (Radians): {', '.join([format_with_uncertainty(i) for i in params_individual_best_dict['i']])}\n"
                                        f"Inclination (i) (Degrees): {', '.join([format_with_uncertainty(i_in_degree) for i_in_degree in params_individual_best_dict['i_in_degree']])}\n"
                                        f"Quadratic Limb Darkening Coefficients (ldc1, ldc2): {', '.join([f'({format_with_uncertainty(ldc1)}, {format_with_uncertainty(ldc2)})' for ldc1, ldc2 in zip(params_individual_best_dict['ldc1'], params_individual_best_dict['ldc2'])])}\n"
                                        f"Transit Duration (Days): {', '.join([format_with_uncertainty(transit_duration) for transit_duration in params_individual_best_dict['transit_duration']])}\n"
                                        f"Transit Duration (Cadences): {', '.join([format_with_uncertainty(transit_duration_in_cadence, precision=0) for transit_duration_in_cadence in params_individual_best_dict['transit_duration_in_cadence']])}\n"
                                        f"Transit Depth: {', '.join([format_with_uncertainty(transit_depth) for transit_depth in params_individual_best_dict['transit_depth']])}\n"
                                        f"Fitting Interation: {', '.join([f'{n_fitting_iteration}' for n_fitting_iteration in params_individual_best_dict['n_fitting_iteration']])}\n"
                                        f"Residual Standard Deviation: {', '.join([f'{residual_std:.6f}' for residual_std in params_individual_best_dict['residual_std']])}\n"
                                        f"Chi-Square: {', '.join([f'{chi_square:.2f}' for chi_square in params_individual_best_dict['chi_square']])}\n"
                                        f"Reduced Chi-Square: {', '.join([f'{reduced_chi_square:.2f}' for reduced_chi_square in params_individual_best_dict['reduced_chi_square']])}\n\n")
else:
    methodology_result_file.write("\n") # write an empty line


methodology_result_file.write("------------------------------Folding And Binning------------------------------\n\n")


methodology_result_file.write(      f"Folded Period (Days): {p_folded}\n"
                                    f"Folded Epoch Time (BTJD): {t0_folded}\n"
                                    f"Folded Light Curve CDPP: {lc_folded_cdpp}\n\n")


methodology_result_file.write(      f"Bin: {bin}\n")
if bin:
    methodology_result_file.write(f"Time Bin Size: {time_bin_size}\n"
                                  f"Binned Light Curve CDPP: {lc_binned_cdpp}\n\n")
else:
    methodology_result_file.write("\n") # write an empty line


if fit_binned:
    methodology_result_file.write("------------------------------------Binned Fitting------------------------------------\n\n")


    methodology_result_file.write(      f"MCMC Steps: {n_steps_binned}\n"
                                        f"Chain Discard Proportion: {chain_discard_proportion_binned}\n\n")


    methodology_result_file.write(      f"{params_binned_best_dict['model']} Model Fitted Binned Corrected Light Curve Parameters: \n"
                                        f"Normalized Planetary Radius (k, R_p/R_s): {format_with_uncertainty(params_binned_best_dict['k'])}\n"
                                        f"Epoch Time (t0): {format_with_uncertainty(params_binned_best_dict['t0'])}\n"
                                        f"Period (p) (fixed to the same as the fitted period): {format_with_uncertainty(params_binned_best_dict['p'])}\n"
                                        f"Normalized Semi-Major Axis (a/R_s): {format_with_uncertainty(params_binned_best_dict['a'])}\n"
                                        f"Inclination (i) (Radians): {format_with_uncertainty(params_binned_best_dict['i'])}\n"
                                        f"Inclination (i) (Degrees): {format_with_uncertainty(params_binned_best_dict['i_in_degree'])}\n"
                                        f"Quadratic Limb Darkening Coefficients (ldc1, ldc2): {format_with_uncertainty(params_binned_best_dict['ldc1'])}, {format_with_uncertainty(params_binned_best_dict['ldc2'])}\n"
                                        f"Transit Duration (Days): {format_with_uncertainty(params_binned_best_dict['transit_duration'])}\n"
                                        f"Transit Duration (Cadences): {format_with_uncertainty(params_binned_best_dict['transit_duration_in_cadence'], precision=0)}\n"
                                        f"Transit Depth: {format_with_uncertainty(params_binned_best_dict['transit_depth'])}\n"
                                        f"Fitting Interation: {params_binned_best_dict['n_fitting_iteration']}\n"
                                        f"Residual Standard Deviation: {params_binned_best_dict['residual_std']:.6f}\n"
                                        f"Chi-Square: {params_binned_best_dict['chi_square']:.2f}\n"
                                        f"Reduced Chi-Square: {params_binned_best_dict['reduced_chi_square']:.2f}\n\n")
else:
    methodology_result_file.write("\n") # write an empty line


methodology_result_file.close()