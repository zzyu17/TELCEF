import os
import warnings

from pytransit import QuadraticModel, QuadraticModelCL, RoadRunnerModel, QPower2Model, GeneralModel
import astropy.units as u
import numpy as np
import corner
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter

from utils import update_dict, format_lc_fits_fn_by_provenance, calculate_cdpp, run_transit_fitting, split_indiviual_lc, plot_trace_evolution, plot_posterior_corner
from A_ab_Configuration_Loader import *




# Define the directories
lc_extracted_dir = data_dir + config['directory']['lc_extracted_dir']
os.makedirs(lc_extracted_dir, exist_ok=True)
lc_extracted_dir_source = lc_extracted_dir + f"/{name}"
os.makedirs(lc_extracted_dir_source, exist_ok=True)

lc_downloaded_dir = data_dir + config['directory']['lc_downloaded_dir']
os.makedirs(lc_downloaded_dir, exist_ok=True)
lc_downloaded_dir_source = lc_downloaded_dir + f"/{name}"
os.makedirs(lc_downloaded_dir_source, exist_ok=True)

eleanor_lc_dir = data_dir + config['directory']['eleanor_lc_dir']
os.makedirs(eleanor_lc_dir, exist_ok=True)
eleanor_lc_dir_source = eleanor_lc_dir + f"/{name}"
os.makedirs(eleanor_lc_dir_source, exist_ok=True)

lightkurve_lc_dir = data_dir + config['directory']['lightkurve_lc_dir']
os.makedirs(lightkurve_lc_dir, exist_ok=True)
lightkurve_lc_dir_source = lightkurve_lc_dir + f"/{name}"
os.makedirs(lightkurve_lc_dir_source, exist_ok=True)

lc_fnb_dir = data_dir + config['directory']['lc_fnb_dir']
os.makedirs(lc_fnb_dir, exist_ok=True)
lc_fnb_dir_source = lc_fnb_dir + f"/{name}"
os.makedirs(lc_fnb_dir_source, exist_ok=True)

pytransit_fitting_plots_dir = base_dir + config["directory"]["pytransit_fitting_plots_dir"]
os.makedirs(pytransit_fitting_plots_dir, exist_ok=True)
pytransit_fitting_plots_dir_source_sector = pytransit_fitting_plots_dir + f"/{name}_Sector-{sector}"
os.makedirs(pytransit_fitting_plots_dir_source_sector, exist_ok=True)




# Define the Lightkurve raw-nans-removed light curve BLS-fitted transit parameters
transit_mask_bls_raw_nans_removed_span_coefficient = config['lightkurve']['transit_mask_bls_raw_nans_removed_span_coefficient'] if config['lightkurve']['transit_mask_bls_raw_nans_removed_span_coefficient'] is not None else 1.8 ##### set the coefficient of BLS transit mask span #####
p_bls_raw_nans_removed = config['lightkurve']['raw_nans_removed_bls_fitted_parameters']['p'] if config['lightkurve']['raw_nans_removed_bls_fitted_parameters']['p'] is not None else None
t0_bls_raw_nans_removed = config['lightkurve']['raw_nans_removed_bls_fitted_parameters']['t0'] if config['lightkurve']['raw_nans_removed_bls_fitted_parameters']['t0'] is not None else None
transit_duration_bls_raw_nans_removed = config['lightkurve']['raw_nans_removed_bls_fitted_parameters']['transit_duration'] if config['lightkurve']['raw_nans_removed_bls_fitted_parameters']['transit_duration'] is not None else None
transit_depth_bls_raw_nans_removed = config['lightkurve']['raw_nans_removed_bls_fitted_parameters']['transit_depth'] if config['lightkurve']['raw_nans_removed_bls_fitted_parameters']['transit_depth'] is not None else None




### ------ Lightcurve Selecting ------ ###
# Set the lightcurve selecting criteria
lc_provenance = config['transit_fitting']['lightcurve_provenance']
lc_fn = format_lc_fits_fn_by_provenance(lc_provenance, config)

# Search for the lightcurve file in the data directory
fits_path_list = [os.path.join(root, f) for root, _, files in os.walk(data_dir) for f in files if f.endswith('.fits')]
fits_fn_list = [os.path.basename(fits_path) for fits_path in fits_path_list]
lc_path = None
for l in range(len(fits_fn_list)):
    if fits_fn_list[l] == lc_fn:
        lc_path = fits_path_list[l]


# Read the lightcurve file via Lightkurve
if lc_path is None:
    raise FileNotFoundError("The specified lightcurve file does not exist in the data directory. Please check if the file exists and run the light curve extracting/downloading scripts first if necessary.")
else:
    lc = lk.read(lc_path).normalize() # Force lightcurve normalization before fitting
    print(f"Successfully found and read the specified lightcurve file: {lc_path}.\n")




# Define the lightcurve-specified directory
lc_fn_pure = os.path.splitext(lc_fn)[0]
lc_fn_suffix = lc_fn_pure.replace(f"{name}_{mission}_Sector-{sector}_", "", 1).replace(f"_LC", "", -1)
correction = config['lightkurve']['correction'] if config['lightkurve']['correction'] is not None else ""

pytransit_fitting_plots_dir_source_sector_lc = pytransit_fitting_plots_dir_source_sector + f"/{lc_fn_suffix}"
os.makedirs(pytransit_fitting_plots_dir_source_sector_lc, exist_ok=True)


# Define the lightcurve plot title
lc_plot_title = lc_fn_pure.replace(f"_{mission}", "", 1).replace(f"{correction}", "").replace("_LC", "", -1).replace("_", " ").replace("Sector-", "Sector ").replace("lightkurve aperture", "lightkurve_aperture")


# Set whether to plot the error bar of the light curve and residuals
plot_errorbar = config["transit_fitting"]["plot_errorbar"]




### ------ Global Fitting ------ ###
i = 1 # count the step


##### Define the global transit fitting parameters #####
fit_global = config['transit_fitting']['fit_global']

# transit fitting parameters
max_iter_global = config['transit_fitting']['max_iter_global'] # maximum number of iterations for the global fitting after removing outliers each time
sigma_global = config['transit_fitting']['sigma_global'] # sigma for the global fitting outliers removal

transit_model_name_global = config['transit_fitting']['transit_model_name_global'] # name of the transit model to use for global fitting
n_walkers_global = config['transit_fitting']['n_walkers_global'] # number of MCMC walkers for global fitting
n_steps_global = config['transit_fitting']['n_steps_global'] # number of MCMC steps for global fitting
chain_discard_proportion_global = config['transit_fitting']['chain_discard_proportion_global'] # the proportion of MCMC chain to discard as burn-in for global fitting

# plotting parameters
# define the default plotting alpha coefficient of the light curve corresponding to the exposure time
if exptime <= 80:
    alpha_exptime_default = 0.1
elif 80 < exptime <= 400:
    alpha_exptime_default = 0.3
elif exptime > 400:
    alpha_exptime_default = 0.5
alpha_exptime = config['transit_fitting']['alpha_exptime'] if config['transit_fitting']['alpha_exptime'] is not None else alpha_exptime_default # the plotting alpha coefficient of the light curve corresponding to the exposure time

# define the default scatter point size coefficient of the light curve corresponding to the exposure time
if exptime <= 80:
    scatter_point_size_exptime_default = 0.05
elif 80 < exptime <= 400:
    scatter_point_size_exptime_default = 0.5
elif exptime > 400:
    scatter_point_size_exptime_default = 1
scatter_point_size_exptime = config['transit_fitting']['scatter_point_size_exptime'] if config['transit_fitting']['scatter_point_size_exptime'] is not None else scatter_point_size_exptime_default # the scatter point size coefficient of the light curve corresponding to the exposure time

chain_thin_global = config['transit_fitting']['chain_thin_global'] # thinning factor of the sample chain when visualizing the process and result for global fitting
running_mean_window_proportion_global = config['transit_fitting']['running_mean_window_proportion_global'] # window length proportion of the thinned-unflattened MCMC chain to calculate the running means of the parameters when visualizing the process and result for global fitting
running_mean_window_length_global = int(running_mean_window_proportion_global * n_steps_global * (1 - chain_discard_proportion_global) / chain_thin_global)


# Set the initial global fitted transit parameters
k_global_initial = np.sqrt(transit_depth_bls_raw_nans_removed) if transit_depth_bls_raw_nans_removed is not None else np.sqrt(transit_depth_nasa) # k: normalized planetary radius, i.e., R_p/R_s
t0_global_initial = t0_bls_raw_nans_removed if t0_bls_raw_nans_removed is not None else t0_nasa # t0: epoch time in BTJD
p_global_initial = p_bls_raw_nans_removed if p_bls_raw_nans_removed is not None else p_nasa # p: orbital period in days
a_global_initial = 10.0 # a: normalized semi-major axis, i.e., a/R_s
i_global_initial = np.pi / 2 # i: orbital inclination in radians
ldc1_global_initial = 0.2 # ldc1: linear limb darkening coefficient
ldc2_global_initial = 0.3 # ldc2: quadratic limb darkening coefficient
params_global_initial = {'k': k_global_initial, 't0': t0_global_initial, 'p': p_global_initial, 'a': a_global_initial, 'i': i_global_initial, 'ldc1': ldc1_global_initial, 'ldc2': ldc2_global_initial}


if fit_global:
    # Run global transit fitting
    results_global = run_transit_fitting(lc, transit_model_name_global, 'global', params_global_initial, n_walkers_global, n_steps_global, chain_discard_proportion_global, chain_thin_global, max_iter_global, sigma_global)
    n_iter_global, n_dim_global, params_global_name, params_global_samples, params_global_samples_thinned_unflattened, params_global_best, params_global_best_lower_error, params_global_best_upper_error, lc, lc_fitted_global, lc_residual_global, residual_std_global, chi_square_global, reduced_chi_square_global = results_global.values()

    config = update_config(config_path, {'transit_fitting.global_fitted_transit_parameters.n_iterations': n_iter_global,

        'transit_fitting.global_fitted_transit_parameters.k': [params_global_best['k'], params_global_best_lower_error['k'], params_global_best_upper_error['k']],
        'transit_fitting.global_fitted_transit_parameters.t0': [params_global_best['t0'], params_global_best_lower_error['t0'], params_global_best_upper_error['t0']],
        'transit_fitting.global_fitted_transit_parameters.p': [params_global_best['p'], params_global_best_lower_error['p'], params_global_best_upper_error['p']],
        'transit_fitting.global_fitted_transit_parameters.a': [params_global_best['a'], params_global_best_lower_error['a'], params_global_best_upper_error['a']],
        'transit_fitting.global_fitted_transit_parameters.i': [params_global_best['i'], params_global_best_lower_error['i'], params_global_best_upper_error['i']],
        'transit_fitting.global_fitted_transit_parameters.i_in_degree': [params_global_best['i_in_degree'], params_global_best_lower_error['i_in_degree'], params_global_best_upper_error['i_in_degree']],
        'transit_fitting.global_fitted_transit_parameters.ldc1': [params_global_best['ldc1'], params_global_best_lower_error['ldc1'], params_global_best_upper_error['ldc1']],
        'transit_fitting.global_fitted_transit_parameters.ldc2': [params_global_best['ldc2'], params_global_best_lower_error['ldc2'], params_global_best_upper_error['ldc2']],
        'transit_fitting.global_fitted_transit_parameters.ldc': [params_global_best['ldc'], params_global_best_lower_error['ldc'], params_global_best_upper_error['ldc']],
        'transit_fitting.global_fitted_transit_parameters.transit_duration': [params_global_best['transit_duration'], params_global_best_lower_error['transit_duration'], params_global_best_upper_error['transit_duration']],
        'transit_fitting.global_fitted_transit_parameters.transit_depth': [params_global_best['transit_depth'], params_global_best_lower_error['transit_depth'], params_global_best_upper_error['transit_depth']],

        'transit_fitting.global_fitted_transit_parameters.residual_std': residual_std_global,
        'transit_fitting.global_fitted_transit_parameters.chi_square': chi_square_global,
        'transit_fitting.global_fitted_transit_parameters.reduced_chi_square': reduced_chi_square_global
    })


    # Plot the visualization plots of the fitting process and result
    # plot the trace and evolution of MCMC parameters
    j = 1 # count the sub-step

    params_global_trace_evolution_plot = plot_trace_evolution(results_global, running_mean_window_length_global)
    params_global_trace_evolution_plot.axes[0].set_title("Trace Of Parameters", fontsize='x-large')
    params_global_trace_evolution_plot.axes[1].set_title(f"Evolution ({running_mean_window_proportion_global * 100}% Window Running Mean) Of Parameters", fontsize='x-large')
    params_global_trace_evolution_plot.suptitle(f"{lc_plot_title} Global Fitted Transit Parameters Trace and Evolution (Thinned By {chain_thin_global})", fontsize='xx-large')
    params_global_trace_evolution_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Global_Fitted_Transit_Parameters_Trace_and_Evolution.png")
    plt.close()


    # plot the MCMC parameters posterior distribution corner plot
    j += 1 # count the sub-step

    params_global_corner_plot = plot_posterior_corner(results_global, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".4f", figsize=(20, 25))
    params_global_corner_plot.suptitle(f"{lc_plot_title} Global Fitted Transit Parameters Posterior Distribution Corner Plot", fontsize='xx-large', y=1.05)
    params_global_corner_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Global_Fitted_Transit_Parameters_Posterior_Distribution_Corner_Plot.png", bbox_inches='tight')
    plt.close()


    # plot the best fitted light curve and residuals
    j += 1 # count the sub-step

    # plot the best fitted model
    lc_fitted_global_plot, (ax_lc_fitted_global, ax_lc_residual_global) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    if plot_errorbar:
        lc.scatter(ax=ax_lc_fitted_global, label=None, s=0.1, alpha=1.0)
        lc.errorbar(ax=ax_lc_fitted_global, label="Original Light Curve", alpha=alpha_exptime)
    else:
        lc.scatter(ax=ax_lc_fitted_global, label="Original Light Curve", s=scatter_point_size_exptime)
    lc_fitted_global.plot(ax=ax_lc_fitted_global, c='red', label=f"Best Fitted {transit_model_name_global} Model, chi-square={chi_square_global:.2f}, reduced chi-square={reduced_chi_square_global:.2f}")
    ax_lc_fitted_global.legend(loc='lower right')
    ax_lc_fitted_global.set_ylabel("Flux")
    ax_lc_fitted_global.tick_params(axis='x', labelbottom=True)
    ax_lc_fitted_global.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    # plot the best fitted model residuals
    if plot_errorbar:
        lc_residual_global.scatter(ax=ax_lc_residual_global, c='green', label=None, s=0.1)
        lc_residual_global.errorbar(ax=ax_lc_residual_global, c='green', label=f"Best Fitted Model Residuals, residual std={residual_std_global:.6f}")
    else:
        lc_residual_global.scatter(ax=ax_lc_residual_global, c='green', label=f"Best Fitted Model Residuals, residual std={residual_std_global:.6f}", s=0.1)
    ax_lc_residual_global.legend(loc='upper right')
    ax_lc_residual_global.set_ylabel("Residuals")
    ax_lc_fitted_global.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    lc_fitted_global_plot.suptitle(f"{lc_plot_title} Global Best Fitted Light Curve and Residuals")
    lc_fitted_global_plot.figure.tight_layout()
    lc_fitted_global_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Global_Best_Fitted_Light_Curve_and_Residuals.png")
    plt.close()




### ------ Individual Fitting ------ ###
i += 1 # count the step


##### Define the individual transit fitting parameters #####
fit_individual = config['transit_fitting']['fit_individual']

# transit fitting parameters
max_iter_individual = config['transit_fitting']['max_iter_individual'] # maximum number of iterations for the individual fitting after removing outliers each time
sigma_individual = config['transit_fitting']['sigma_individual'] # sigma for the individual fitting outlier removal

transit_model_name_individual = config['transit_fitting']['transit_model_name_individual'] # name of the transit model to use for individual fitting
n_walkers_individual = config['transit_fitting']['n_walkers_individual'] # number of MCMC walkers for individual fitting
n_steps_individual = config['transit_fitting']['n_steps_individual'] # number of MCMC steps for individual fitting
chain_discard_proportion_individual = config['transit_fitting']['chain_discard_proportion_individual'] # the proportion of MCMC chain to discard as burn-in for individual fitting

individual_transit_check_coefficient = config['transit_fitting']['individual_transit_check_coefficient'] # the coefficient of transit duration span to check if the individual transit light curve contains transit event

# plotting parameters
chain_thin_individual = config['transit_fitting']['chain_thin_individual'] # thinning factor of the sample chain when visualizing the process and result for individual fitting
running_mean_window_proportion_individual = config['transit_fitting']['running_mean_window_proportion_individual'] # window length proportion of the thinned-unflattened MCMC chain to calculate the running means of the parameters when visualizing the process and result for individual fitting
running_mean_window_length_individual = int(running_mean_window_proportion_individual * n_steps_individual * (1 - chain_discard_proportion_individual) / chain_thin_individual)
individual_transit_plot_coefficient = config['transit_fitting']['individual_transit_plot_coefficient'] # the coefficient of the individual transit plot span


# Set the initial individual fitted transit parameters
k_individual_initial = params_global_best['k'] if fit_global else k_global_initial # k: normalized planetary radius, i.e., R_p/R_s
t0_individual = params_global_best['t0'] if fit_global else t0_global_initial # t0: epoch time in BTJD
p_individual = params_global_best['p'] if fit_global else p_global_initial # p: orbital period in days
a_individual_initial = params_global_best['a'] if fit_global else 10.0 # a: normalized semi-major axis, i.e., a/R_s
i_individual_initial = params_global_best['i'] if fit_global else np.pi / 2 # i: orbital inclination in radians
ldc1_individual_initial = params_global_best['ldc1'] if fit_global else 0.2 # ldc1: linear limb darkening coefficient
ldc2_individual_initial = params_global_best['ldc2'] if fit_global else 0.3 # ldc2: quadratic limb darkening coefficient
params_individual_initial = {'k': k_individual_initial, 't0': t0_individual, 'p': p_individual, 'a': a_individual_initial, 'i': i_individual_initial, 'ldc1': ldc1_individual_initial, 'ldc2': ldc2_individual_initial}

if fit_global:
    transit_duration = params_global_best['transit_duration']
elif transit_duration_bls_raw_nans_removed is not None:
    transit_duration = transit_duration_bls_raw_nans_removed * transit_mask_bls_raw_nans_removed_span_coefficient
else:
    transit_duration = transit_duration_nasa


if fit_individual:
    # Split the light curve into individual transit light curves and check if they contain transit events
    lc_individual_list, transit_info = split_indiviual_lc(lc, p_individual, t0_individual, transit_duration, individual_transit_check_coefficient)
    n_possible_transits = transit_info['n_possible_transits']
    n_valid_transits = transit_info['n_valid_transits']
    no_data_transit_indices = transit_info['no_data_transit_indices']
    valid_transit_indices = transit_info['valid_transit_indices']

    lc_fitted_individual_list = [None] * n_possible_transits
    lc_residual_individual_list = [None] * n_possible_transits
    individual_transit_plot_mask_list = [None] * n_possible_transits

    config = update_config(config_path, {
        'transit_fitting.individual_fitted_transit_parameters.n_possible_transits': n_possible_transits,
        'transit_fitting.individual_fitted_transit_parameters.n_valid_transits': n_valid_transits,
        'transit_fitting.individual_fitted_transit_parameters.no_data_transit_indices': no_data_transit_indices,
        'transit_fitting.individual_fitted_transit_parameters.valid_transit_indices': valid_transit_indices,

        'transit_fitting.individual_fitted_transit_parameters.n_iterations': [None] * n_possible_transits,

        'transit_fitting.individual_fitted_transit_parameters.k': [[None, None, None] for transit_index in range(n_possible_transits)],
        'transit_fitting.individual_fitted_transit_parameters.t0': [[None, None, None] for transit_index in range(n_possible_transits)],
        'transit_fitting.individual_fitted_transit_parameters.p': [None, None, None], # period is fixed for individual transit fitting
        'transit_fitting.individual_fitted_transit_parameters.a': [[None, None, None] for transit_index in range(n_possible_transits)],
        'transit_fitting.individual_fitted_transit_parameters.i': [[None, None, None] for transit_index in range(n_possible_transits)],
        'transit_fitting.individual_fitted_transit_parameters.i_in_degree': [[None, None, None] for transit_index in range(n_possible_transits)],
        'transit_fitting.individual_fitted_transit_parameters.ldc1': [[None, None, None] for transit_index in range(n_possible_transits)],
        'transit_fitting.individual_fitted_transit_parameters.ldc2': [[None, None, None] for transit_index in range(n_possible_transits)],
        'transit_fitting.individual_fitted_transit_parameters.ldc': [[[None, None], [None, None], [None, None]] for transit_index in range(n_possible_transits)],
        'transit_fitting.individual_fitted_transit_parameters.transit_duration': [[None, None, None] for transit_index in range(n_possible_transits)],
        'transit_fitting.individual_fitted_transit_parameters.transit_depth': [[None, None, None] for transit_index in range(n_possible_transits)],

        'transit_fitting.individual_fitted_transit_parameters.residual_std': [None] * n_possible_transits,
        'transit_fitting.individual_fitted_transit_parameters.chi_square': [None] * n_possible_transits,
        'transit_fitting.individual_fitted_transit_parameters.reduced_chi_square': [None] * n_possible_transits
    })


    for transit_index in range(n_possible_transits):
        if transit_index in valid_transit_indices:
            lc_individual = lc_individual_list[transit_index]
            t0_individual_initial = t0_individual + p_individual * transit_index # update t0 for each individual transit fitting
            params_individual_initial = update_dict(params_individual_initial, {'t0': t0_individual_initial})

            # Run individual transit fitting
            results_individual = run_transit_fitting(lc_individual, transit_model_name_individual, 'individual', params_individual_initial, n_walkers_individual, n_steps_individual, chain_discard_proportion_individual, chain_thin_individual, max_iter_individual, sigma_individual, transit_index=transit_index)
            n_iter_individual, n_dim_individual, params_individual_name, params_individual_samples, params_individual_samples_thinned_unflattened, params_individual_best, params_individual_best_lower_error, params_individual_best_upper_error, lc_individual, lc_fitted_individual, lc_residual_individual, residual_std_individual, chi_square_individual, reduced_chi_square_individual = results_individual.values()

            # Store the individual light curve, the individual best fitted light curve and residuals into the lists
            lc_individual_list[transit_index] = lc_individual
            lc_fitted_individual_list[transit_index] = lc_fitted_individual
            lc_residual_individual_list[transit_index] = lc_residual_individual

            config = update_config(config_path, {
                f'transit_fitting.individual_fitted_transit_parameters.n_iterations[{transit_index}]': n_iter_individual,

                f'transit_fitting.individual_fitted_transit_parameters.k[{transit_index}]': [params_individual_best['k'], params_individual_best_lower_error['k'], params_individual_best_upper_error['k']],
                f'transit_fitting.individual_fitted_transit_parameters.t0[{transit_index}]': [params_individual_best['t0'], params_individual_best_lower_error['t0'], params_individual_best_upper_error['t0']],
                f'transit_fitting.individual_fitted_transit_parameters.a[{transit_index}]': [params_individual_best['a'], params_individual_best_lower_error['a'], params_individual_best_upper_error['a']],
                f'transit_fitting.individual_fitted_transit_parameters.i[{transit_index}]': [params_individual_best['i'], params_individual_best_lower_error['i'], params_individual_best_upper_error['i']],
                f'transit_fitting.individual_fitted_transit_parameters.i_in_degree[{transit_index}]': [params_individual_best['i_in_degree'], params_individual_best_lower_error['i_in_degree'], params_individual_best_upper_error['i_in_degree']],
                f'transit_fitting.individual_fitted_transit_parameters.ldc1[{transit_index}]': [params_individual_best['ldc1'], params_individual_best_lower_error['ldc1'], params_individual_best_upper_error['ldc1']],
                f'transit_fitting.individual_fitted_transit_parameters.ldc2[{transit_index}]': [params_individual_best['ldc2'], params_individual_best_lower_error['ldc2'], params_individual_best_upper_error['ldc2']],
                f'transit_fitting.individual_fitted_transit_parameters.ldc[{transit_index}]': [params_individual_best['ldc'], params_individual_best_lower_error['ldc'], params_individual_best_upper_error['ldc']],
                f'transit_fitting.individual_fitted_transit_parameters.transit_duration[{transit_index}]': [params_individual_best['transit_duration'], params_individual_best_lower_error['transit_duration'], params_individual_best_upper_error['transit_duration']],
                f'transit_fitting.individual_fitted_transit_parameters.transit_depth[{transit_index}]': [params_individual_best['transit_depth'], params_individual_best_lower_error['transit_depth'], params_individual_best_upper_error['transit_depth']],

                f'transit_fitting.individual_fitted_transit_parameters.residual_std[{transit_index}]': residual_std_individual,
                f'transit_fitting.individual_fitted_transit_parameters.chi_square[{transit_index}]': chi_square_individual,
                f'transit_fitting.individual_fitted_transit_parameters.reduced_chi_square[{transit_index}]': reduced_chi_square_individual
            })

            # Update the period parameter only at the first valid transit fitting
            if transit_index == valid_transit_indices[0]:
                config = update_config(config_path, {f'transit_fitting.individual_fitted_transit_parameters.p': [params_individual_best['p'], params_individual_best_lower_error['p'], params_individual_best_upper_error['p']]}) # period is fixed for individual transit fitting


            # Plot the visualization plots of the fitting process and result
            # define the directories
            pytransit_fitting_plots_dir_source_sector_lc_transit = pytransit_fitting_plots_dir_source_sector_lc + f"/Transit {transit_index:02}"
            os.makedirs(pytransit_fitting_plots_dir_source_sector_lc_transit, exist_ok=True)


            # plot the trace and evolution of MCMC parameters
            j = 1 # count the sub-step

            params_individual_trace_evolution_plot = plot_trace_evolution(results_individual, running_mean_window_length_individual)
            params_individual_trace_evolution_plot.axes[0].set_title("Trace Of Parameters", fontsize='x-large')
            params_individual_trace_evolution_plot.axes[1].set_title(f"Evolution ({running_mean_window_proportion_individual * 100}% Window Running Mean) Of Parameters", fontsize='x-large')
            params_individual_trace_evolution_plot.suptitle(f"{lc_plot_title} Individual Transit {transit_index:02} Fitted Transit Parameters Trace and Evolution Plot (Thinned By {chain_thin_individual})", fontsize='xx-large')
            params_individual_trace_evolution_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc_transit + f"/{i:02}-{j:01}-{transit_index:02}_Individual_Transit-{transit_index:02}_Fitted_Transit_Parameters_Trace_and_Evolution.png")
            plt.close()


            # plot the MCMC parameters posterior distribution corner plot
            j += 1 # count the sub-step

            params_individual_corner_plot = plot_posterior_corner(results_individual, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".4f", figsize=(20, 25))
            params_individual_corner_plot.suptitle(f"{lc_plot_title} Individual Transit {transit_index:02} Fitted Transit Parameters Posterior Distribution Corner Plot", fontsize='xx-large', y=1.05)
            params_individual_corner_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc_transit + f"/{i:02}-{j:01}-{transit_index:02}_Individual_Transit-{transit_index:02}_Fitted_Transit_Parameters_Posterior_Distribution_Corner_Plot.png", bbox_inches='tight')
            plt.close()


            # plot the best fitted light curve and residuals
            j += 1 # count the sub-step

            # create the individual transit plot mask, store it into the list and apply it to the lightcurves
            individual_transit_plot_range = (params_individual_best['t0'] - params_individual_best['transit_duration'] / 2 * individual_transit_plot_coefficient, params_individual_best['t0'] + params_individual_best['transit_duration'] / 2 * individual_transit_plot_coefficient)
            individual_transit_plot_mask = ((lc_individual.time.value >= individual_transit_plot_range[0]) & (lc_individual.time.value < individual_transit_plot_range[1]))
            individual_transit_plot_mask_list[transit_index] = individual_transit_plot_mask
            lc_individual_masked = lc_individual[individual_transit_plot_mask]
            lc_fitted_individual_masked = lc_fitted_individual[individual_transit_plot_mask]
            lc_residual_individual_masked = lc_residual_individual[individual_transit_plot_mask]

            # plot the best fitted model
            lc_fitted_individual_plot, (ax_lc_fitted_individual, ax_lc_residual_individual) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
            if plot_errorbar:
                lc_individual_masked.scatter(ax=ax_lc_fitted_individual, label=None, s=0.1, alpha=1.0)
                lc_individual_masked.errorbar(ax=ax_lc_fitted_individual, label="Original Light Curve", alpha=alpha_exptime * 5 if alpha_exptime <= 1/5 else 1.0)
            else:
                lc_individual_masked.scatter(ax=ax_lc_fitted_individual, label="Original Light Curve", s=scatter_point_size_exptime * 6)
            lc_fitted_individual_masked.plot(ax=ax_lc_fitted_individual, c='red', label=f"Best Fitted {transit_model_name_individual} Model, chi-square={chi_square_individual:.2f}, reduced chi-square={reduced_chi_square_individual:.2f}")
            ax_lc_fitted_individual.legend(loc='lower right')
            ax_lc_fitted_individual.set_ylabel("Flux")
            ax_lc_fitted_individual.tick_params(axis='x', labelbottom=True)
            ax_lc_fitted_individual.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax_lc_fitted_individual.set_xlim(individual_transit_plot_range[0], individual_transit_plot_range[1])
            # plot the best fitted model residuals
            if plot_errorbar:
                lc_residual_individual_masked.scatter(ax=ax_lc_residual_individual, c='green', label=None, s=0.1)
                lc_residual_individual_masked.errorbar(ax=ax_lc_residual_individual, c='green', label=f"Best Fitted Model Residuals, residual std={residual_std_individual:.6f}")
            else:
                lc_residual_individual_masked.scatter(ax=ax_lc_residual_individual, c='green', label=f"Best Fitted Model Residuals, residual std={residual_std_individual:.6f}", s=0.1)
            ax_lc_residual_individual.legend(loc='upper right')
            ax_lc_residual_individual.set_ylabel("Residuals")
            ax_lc_residual_individual.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax_lc_residual_individual.set_xlim(individual_transit_plot_range[0], individual_transit_plot_range[1])

            lc_fitted_individual_plot.suptitle(f"{lc_plot_title} Individual Transit {transit_index:02} Best Fitted Light Curve and Residuals")
            lc_fitted_individual_plot.figure.tight_layout()
            lc_fitted_individual_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc_transit + f"/{i:02}-{j:01}-{transit_index:02}_Individual_Transit-{transit_index:02}_Best_Fitted_Light_Curve_and_Residuals.png")
            plt.close()


    # plot the all-in-one best fitted light curves
    lc_fitted_individual_all_plot, axes_lc_fitted_individual = plt.subplots(n_valid_transits, 1, figsize=(20, 5 * n_valid_transits))
    if n_valid_transits == 1:
        axes_lc_fitted_individual = [axes_lc_fitted_individual]

    for transit_plot_index, transit_index in enumerate(valid_transit_indices):
        individual_transit_plot_range = (config['transit_fitting']['individual_fitted_transit_parameters']['t0'][transit_index][0] - config['transit_fitting']['individual_fitted_transit_parameters']['transit_duration'][transit_index][0] / 2 * individual_transit_plot_coefficient,
                                         config['transit_fitting']['individual_fitted_transit_parameters']['t0'][transit_index][0] + config['transit_fitting']['individual_fitted_transit_parameters']['transit_duration'][transit_index][0] / 2 * individual_transit_plot_coefficient)
        individual_transit_plot_mask = individual_transit_plot_mask_list[transit_index]
        lc_individual_masked = lc_individual_list[transit_index][individual_transit_plot_mask]
        lc_fitted_individual_masked = lc_fitted_individual_list[transit_index][individual_transit_plot_mask]

        ax_lc_fitted_individual = axes_lc_fitted_individual[transit_plot_index]
        if plot_errorbar:
            lc_individual_masked.scatter(ax=ax_lc_fitted_individual, label=None, s=0.1, alpha=1.0)
            lc_individual_masked.errorbar(ax=ax_lc_fitted_individual, label="Original Light Curve" if transit_plot_index == 0 else None, alpha=alpha_exptime * 5 if alpha_exptime <= 1 / 5 else 1.0)
        else:
            lc_individual_masked.scatter(ax=ax_lc_fitted_individual, label="Original Light Curve" if transit_plot_index == 0 else None, s=scatter_point_size_exptime * 6)
        lc_fitted_individual_masked.plot(ax=ax_lc_fitted_individual, c='red', label=f"Best Fitted {transit_model_name_individual} Model, chi-square={config['transit_fitting']['individual_fitted_transit_parameters']['chi_square'][transit_index]:.2f},\n"
                                                                                    f"reduced chi-square={config['transit_fitting']['individual_fitted_transit_parameters']['reduced_chi_square'][transit_index]:.2f}, residual std={config['transit_fitting']['individual_fitted_transit_parameters']['residual_std'][transit_index]:.6f}" if transit_plot_index == 0
                                                                               else f"chi-square={config['transit_fitting']['individual_fitted_transit_parameters']['chi_square'][transit_index]:.2f}, reduced chi-square={config['transit_fitting']['individual_fitted_transit_parameters']['reduced_chi_square'][transit_index]:.2f},\n"
                                                                                    f"residual std={config['transit_fitting']['individual_fitted_transit_parameters']['residual_std'][transit_index]:.6f}")
        ax_lc_fitted_individual.legend(loc='lower right')
        ax_lc_fitted_individual.set_ylabel("Flux")
        ax_lc_fitted_individual.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax_lc_fitted_individual.set_xlim(individual_transit_plot_range[0], individual_transit_plot_range[1])
        ax_lc_fitted_individual.set_title(f"Transit {transit_index:02}", fontsize='x-large')

    lc_fitted_individual_all_plot.suptitle(f"{lc_plot_title} All-in-one Individual Best Fitted Light Curves", fontsize='xx-large', y=1.00)
    lc_fitted_individual_all_plot.tight_layout()
    lc_fitted_individual_all_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}_All-in-one_Individual_Best_Fitted_Light_Curves.png", bbox_inches='tight')
    plt.close()




##### Define the folding and binning parameters #####
fold = config['transit_fitting']['fold']
p_fold = config['transit_fitting']['p_fold'] if config['transit_fitting']['p_fold'] is not None else p_individual
t0_fold = config['transit_fitting']['t0_fold'] if config['transit_fitting']['t0_fold'] is not None else t0_individual

bin = config['transit_fitting']['bin']
time_bin_size = config['transit_fitting']['time_bin_size'] * u.second if config['transit_fitting']['time_bin_size'] is not None else exptime * u.second




### ------ Fold ------ ###
# Fold the lightcurve based on the fitted parameters and plot the folded light curve
i += 1 # count the step


if fold:
    lc_folded = lc.fold(period=p_fold, epoch_time=t0_fold)
    lc_folded_cdpp = calculate_cdpp(lc_folded, exptime=exptime)

    lc_folded_plot, ax_lc_folded = plt.subplots(figsize=(10, 5))
    if plot_errorbar:
        lc_folded.scatter(ax=ax_lc_folded, label=None, s=0.1, alpha=1.0)
        lc_folded.errorbar(ax=ax_lc_folded, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_folded_cdpp:.2f} ppm", alpha=alpha_exptime / 2)
    else:
        lc_folded.scatter(ax=ax_lc_folded, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_folded_cdpp:.2f} ppm", s=scatter_point_size_exptime)
    ax_lc_folded.legend(loc='lower right')
    ax_lc_folded.set_title(f"{lc_plot_title} Period={p_fold:.4f}d Folded Light Curve")
    ax_lc_folded.set_ylabel("Flux")
    lc_folded_plot.figure.tight_layout()
    lc_folded_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}_Period={p_fold:.4f}d_Folded_Light_Curve.png")
    plt.close()


else:
    lc_folded = lc.copy()
    lc_folded_cdpp = calculate_cdpp(lc_folded, exptime=exptime)




### ------ Bin ------ ###
# Bin the lightcurve and plot the binned light curve
i += 1 # count the step


if bin:
    lc_fnb = lc_folded.bin(time_bin_size=time_bin_size)
    lc_fnb_cdpp = calculate_cdpp(lc_fnb, exptime=exptime)

    lc_fnb_plot, ax_lc_fnb = plt.subplots(figsize=(10, 5))
    if plot_errorbar:
        lc_fnb.scatter(ax=ax_lc_fnb, label=None, s=0.1, alpha=1.0)
        lc_fnb.errorbar(ax=ax_lc_fnb, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_fnb_cdpp:.2f} ppm", alpha=alpha_exptime * 5 if alpha_exptime <= 1/5 else 1.0)
    else:
        lc_fnb.scatter(ax=ax_lc_fnb, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_fnb_cdpp:.2f} ppm", s=scatter_point_size_exptime * 6)
    ax_lc_fnb.legend(loc='lower right')
    if fold:
        ax_lc_fnb.set_title(f"{lc_plot_title} Time-bin-size={time_bin_size:.1f} Binned\nPeriod={p_fold:.4f}d Folded Light Curve")
        ax_lc_fnb.set_ylabel("Flux")
        lc_fnb_plot.figure.tight_layout()
        lc_fnb_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}_Time-Bin-Size={time_bin_size.value}s_Binned_Period={p_fold:.4f}d_Folded_Light_Curve.png")
    else:
        ax_lc_fnb.set_title(f"{lc_plot_title} Time-bin-size={time_bin_size:.1f} Binned Light Curve")
        ax_lc_fnb.set_ylabel("Flux")
        lc_fnb_plot.figure.tight_layout()
        lc_fnb_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}_Time-Bin-Size={time_bin_size.value}s_Binned_Light_Curve.png")
    plt.close()


else:
    lc_fnb = lc_folded.copy()
    lc_fnb_cdpp = calculate_cdpp(lc_fnb, exptime=exptime)




# Save the folded-and-binned lightcurve into a FITS file
if fold and bin:
    fnb = f"_Folded-p{p_fold:.4f}-t{t0_fold:.4f}_&_Binned-t{time_bin_size.value:.1f}"
elif fold and not bin:
    fnb = f"_Folded-p{p_fold:.4f}-t{t0_fold:.4f}"
elif not fold and bin:
    fnb = f"_Binned-t{time_bin_size.value:.1f}"
else:
    fnb = ""

config = update_config(config_path, {'transit_fitting.fnb': fnb})

lc_fnb_fn = lc_fn.replace("_LC", f"{fnb}_LC", -1)
lc_fnb_path = lc_fnb_dir_source + f"/{lc_fnb_fn}"
lc_fnb.to_fits(path=lc_fnb_path, overwrite=True)


print(f"Successfully folded and/or binned the light curve and saved it to the data directory of the source: {lc_fnb_path}.\n")




### ------ Folded-and-binned Fitting ------ ###
i += 1 # count the step


##### Define the folded-and-binned transit fitting parameters #####
fit_fnb = config['transit_fitting']['fit_fnb']

# transit fitting parameters
max_iter_fnb = config['transit_fitting']['max_iter_fnb'] # maximum number of iterations for the folded-and-binned fitting after removing outliers each time
sigma_fnb = config['transit_fitting']['sigma_fnb'] # sigma for the folded-and-binned fitting outliers removal

transit_model_name_fnb = config['transit_fitting']['transit_model_name_fnb'] # name of the transit model to use for folded-and-binned fitting
n_walkers_fnb = config['transit_fitting']['n_walkers_fnb'] # number of MCMC walkers for folded-and-binned fitting
n_steps_fnb = config['transit_fitting']['n_steps_fnb'] # number of MCMC steps for folded-and-binned fitting
chain_discard_proportion_fnb = config['transit_fitting']['chain_discard_proportion_fnb'] # the proportion of MCMC chain to discard as burn-in for folded-and-binned fitting

# plotting parameters
chain_thin_fnb = config['transit_fitting']['chain_thin_fnb'] # thinning factor of the sample chain when visualizing the process and result for folded-and-binned fitting
running_mean_window_proportion_fnb = config['transit_fitting']['running_mean_window_proportion_fnb'] # window length proportion of the thinned-unflattened MCMC chain to calculate the running means of the parameters when visualizing the process and result for folded-and-binned fitting
running_mean_window_length_fnb = int(running_mean_window_proportion_fnb * n_steps_fnb * (1 - chain_discard_proportion_fnb) / chain_thin_fnb)
folded_transit_plot_coefficient = config['transit_fitting']['folded_transit_plot_coefficient'] # the coefficient of the folded-and-binned transit plot span, only used when 'fold' is set to true (i.e., the lightcurve is folded)


# Set the initial folded-and-binned fitted transit parameters
k_fnb_initial = params_global_best['k'] if fit_global else k_global_initial # k: normalized planetary radius, i.e., R_p/R_s
p_fnb_initial = params_global_best['p'] if fit_global else p_global_initial # p: orbital period in days
a_fnb_initial = params_global_best['a'] if fit_global else a_global_initial # a: normalized semi-major axis, i.e., a/R_s
i_fnb_initial = params_global_best['i'] if fit_global else i_global_initial # i: orbital inclination in radians
ldc1_fnb_initial = params_global_best['ldc1'] if fit_global else ldc1_global_initial # ldc1: linear limb darkening coefficient
ldc2_fnb_initial = params_global_best['ldc2'] if fit_global else ldc2_global_initial # ldc2: quadratic limb darkening coefficient
if fold:
    t0_fnb_initial = 0.0
else:
    t0_fnb_initial = params_global_best['t0'] if fit_global else t0_global_initial # t0: epoch time
params_fnb_initial = {'k': k_fnb_initial, 't0': t0_fnb_initial, 'p': p_fnb_initial, 'a': a_fnb_initial, 'i': i_fnb_initial, 'ldc1': ldc1_fnb_initial, 'ldc2': ldc2_fnb_initial}


if fit_fnb:
    # Run folded-and-binned transit fitting
    results_fnb = run_transit_fitting(lc_fnb, transit_model_name_fnb, 'folded' if fold else 'global', params_fnb_initial, n_walkers_fnb, n_steps_fnb, chain_discard_proportion_fnb, chain_thin_fnb, max_iter_fnb, sigma_fnb)
    n_iter_fnb, n_dim_fnb, params_fnb_name, params_fnb_samples, params_fnb_samples_thinned_unflattened, params_fnb_best, params_fnb_best_lower_error, params_fnb_best_upper_error, lc_fnb, lc_fitted_fnb, lc_residual_fnb, residual_std_fnb, chi_square_fnb, reduced_chi_square_fnb = results_fnb.values()

    config = update_config(config_path, {
        'transit_fitting.fnb_fitted_transit_parameters.n_iterations': n_iter_fnb,

        'transit_fitting.fnb_fitted_transit_parameters.k': [params_fnb_best['k'], params_fnb_best_lower_error['k'], params_fnb_best_upper_error['k']],
        'transit_fitting.fnb_fitted_transit_parameters.t0': [params_fnb_best['t0'], params_fnb_best_lower_error['t0'], params_fnb_best_upper_error['t0']],
        'transit_fitting.fnb_fitted_transit_parameters.p': [params_fnb_best['p'], params_fnb_best_lower_error['p'], params_fnb_best_upper_error['p']],
        'transit_fitting.fnb_fitted_transit_parameters.a': [params_fnb_best['a'], params_fnb_best_lower_error['a'], params_fnb_best_upper_error['a']],
        'transit_fitting.fnb_fitted_transit_parameters.i': [params_fnb_best['i'], params_fnb_best_lower_error['i'], params_fnb_best_upper_error['i']],
        'transit_fitting.fnb_fitted_transit_parameters.i_in_degree': [params_fnb_best['i_in_degree'], params_fnb_best_lower_error['i_in_degree'], params_fnb_best_upper_error['i_in_degree']],
        'transit_fitting.fnb_fitted_transit_parameters.ldc1': [params_fnb_best['ldc1'], params_fnb_best_lower_error['ldc1'], params_fnb_best_upper_error['ldc1']],
        'transit_fitting.fnb_fitted_transit_parameters.ldc2': [params_fnb_best['ldc2'], params_fnb_best_lower_error['ldc2'], params_fnb_best_upper_error['ldc2']],
        'transit_fitting.fnb_fitted_transit_parameters.ldc': [params_fnb_best['ldc'], params_fnb_best_lower_error['ldc'], params_fnb_best_upper_error['ldc']],
        'transit_fitting.fnb_fitted_transit_parameters.transit_duration': [params_fnb_best['transit_duration'], params_fnb_best_lower_error['transit_duration'], params_fnb_best_upper_error['transit_duration']],
        'transit_fitting.fnb_fitted_transit_parameters.transit_depth': [params_fnb_best['transit_depth'], params_fnb_best_lower_error['transit_depth'], params_fnb_best_upper_error['transit_depth']],

        'transit_fitting.fnb_fitted_transit_parameters.residual_std': residual_std_fnb,
        'transit_fitting.fnb_fitted_transit_parameters.chi_square': chi_square_fnb,
        'transit_fitting.fnb_fitted_transit_parameters.reduced_chi_square': reduced_chi_square_fnb
    })


    # Plot the visualization plots of the fitting process and result
    # plot the trace and evolution of MCMC parameters
    j = 1  # count the sub-step

    params_fnb_trace_evolution_plot = plot_trace_evolution(results_fnb, running_mean_window_length_fnb)
    params_fnb_trace_evolution_plot.axes[0].set_title("Trace Of Parameters", fontsize='x-large')
    params_fnb_trace_evolution_plot.axes[1].set_title(f"Evolution ({running_mean_window_proportion_fnb * 100}% Window Running Mean) Of Parameters", fontsize='x-large')
    if fold and bin:
        params_fnb_trace_evolution_plot.suptitle(f"{lc_plot_title} Folded-and-binned Fitted Transit Parameters Trace and Evolution (Thinned By {chain_thin_fnb})", fontsize='xx-large')
        params_fnb_trace_evolution_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Folded-and-binned_Fitted_Transit_Parameters_Trace_and_Evolution.png")
    elif fold and not bin:
        params_fnb_trace_evolution_plot.suptitle(f"{lc_plot_title} Folded Fitted Transit Parameters Trace and Evolution (Thinned By {chain_thin_fnb})", fontsize='xx-large')
        params_fnb_trace_evolution_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Folded_Fitted_Transit_Parameters_Trace_and_Evolution.png")
    elif bin and not fold:
        params_fnb_trace_evolution_plot.suptitle(f"{lc_plot_title} Binned Fitted Transit Parameters Trace and Evolution (Thinned By {chain_thin_fnb})", fontsize='xx-large')
        params_fnb_trace_evolution_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Binned_Fitted_Transit_Parameters_Trace_and_Evolution.png")
    plt.close()


    # plot the MCMC parameters posterior distribution corner plot
    j += 1  # count the sub-step

    params_fnb_corner_plot = plot_posterior_corner(results_fnb, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".4f", figsize=(20, 25))
    if fold and bin:
        params_fnb_corner_plot.suptitle(f"{lc_plot_title} Folded-and-binned Fitted Transit Parameters Posterior Distribution Corner Plot", fontsize='xx-large', y=1.05)
        params_fnb_corner_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Folded-and-binned_Fitted_Transit_Parameters_Posterior_Distribution_Corner_Plot.png", bbox_inches='tight')
    elif fold and not bin:
        params_fnb_corner_plot.suptitle(f"{lc_plot_title} Folded Fitted Transit Parameters Posterior Distribution Corner Plot", fontsize='xx-large', y=1.05)
        params_fnb_corner_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Folded_Fitted_Transit_Parameters_Posterior_Distribution_Corner_Plot.png", bbox_inches='tight')
    elif bin and not fold:
        params_fnb_corner_plot.suptitle(f"{lc_plot_title} Binned Fitted Transit Parameters Posterior Distribution Corner Plot", fontsize='xx-large', y=1.05)
        params_fnb_corner_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Binned_Fitted_Transit_Parameters_Posterior_Distribution_Corner_Plot.png", bbox_inches='tight')
    plt.close()


    # plot the best fitted light curve and residuals
    j += 1 # count the sub-step

    # plot the best fitted model
    lc_fitted_fnb_plot, (ax_lc_fitted_fnb, ax_lc_residual_fnb) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    if plot_errorbar:
        lc_fnb.scatter(ax=ax_lc_fitted_fnb, label=None, s=0.1, alpha=1.0)
        if fold and bin:
            lc_fnb.errorbar(ax=ax_lc_fitted_fnb, label="Original Light Curve", alpha=alpha_exptime * 5 if alpha_exptime <= 1/5 else 1.0)
        elif fold and not bin:
            lc_fnb.errorbar(ax=ax_lc_fitted_fnb, label="Original Light Curve", alpha=alpha_exptime / 2)
        elif bin and not fold:
            lc_fnb.errorbar(ax=ax_lc_fitted_fnb, label="Original Light Curve", alpha=1.0)
    else:
        if fold and bin:
            lc_fnb.scatter(ax=ax_lc_fitted_fnb, label="Original Light Curve", s=scatter_point_size_exptime * 6)
        elif fold and not bin:
            lc_fnb.scatter(ax=ax_lc_fitted_fnb, label="Original Light Curve", s=scatter_point_size_exptime)
        elif bin and not fold:
            lc_fnb.scatter(ax=ax_lc_fitted_fnb, label="Original Light Curve", s=scatter_point_size_exptime * 6)
    lc_fitted_fnb.plot(ax=ax_lc_fitted_fnb, c='red', label=f"Best Fitted {transit_model_name_fnb} Model, chi-square={chi_square_fnb:.2f}, reduced chi-square={reduced_chi_square_fnb:.2f}")
    ax_lc_fitted_fnb.legend(loc='lower right')
    ax_lc_fitted_fnb.set_ylabel("Flux")
    ax_lc_fitted_fnb.tick_params(axis='x', labelbottom=True)
    ax_lc_fitted_fnb.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    # plot the best fitted model residuals
    if plot_errorbar:
        lc_residual_fnb.scatter(ax=ax_lc_residual_fnb, c='green', label=None, s=0.1)
        lc_residual_fnb.errorbar(ax=ax_lc_residual_fnb, c='green', label=f"Best Fitted Model Residuals, residual std={residual_std_fnb:.6f}")
    else:
        lc_residual_fnb.scatter(ax=ax_lc_residual_fnb, c='green', label=f"Best Fitted Model Residuals, residual std={residual_std_fnb:.6f}", s=0.1)
    ax_lc_residual_fnb.legend(loc='upper right')
    ax_lc_residual_fnb.set_ylabel("Residuals")
    ax_lc_fitted_fnb.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    if fold and bin:
        lc_fitted_fnb_plot.suptitle(f"{lc_plot_title} Folded-and-binned Best Fitted Light Curve and Residuals")
        lc_fitted_fnb_plot.figure.tight_layout()
        lc_fitted_fnb_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Folded-and-binned_Best_Fitted_Light_Curve_and_Residuals.png")
    elif fold and not bin:
        lc_fitted_fnb_plot.suptitle(f"{lc_plot_title} Folded Best Fitted Light Curve and Residuals")
        lc_fitted_fnb_plot.figure.tight_layout()
        lc_fitted_fnb_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Folded_Best_Fitted_Light_Curve_and_Residuals.png")
    elif bin and not fold:
        lc_fitted_fnb_plot.suptitle(f"{lc_plot_title} Binned Best Fitted Light Curve and Residuals")
        lc_fitted_fnb_plot.figure.tight_layout()
        lc_fitted_fnb_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Binned_Best_Fitted_Light_Curve_and_Residuals.png")
    plt.close()