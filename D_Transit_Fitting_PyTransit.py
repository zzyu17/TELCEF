import os
import warnings

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from utils import (update_dict, format_lc_fits_fn_by_provenance, calculate_cdpp, alpha_exptime_default, scatter_point_size_exptime_default,
                   PARAMS_NAME, load_params_initial_from_config, load_priors_from_config, load_params_fold_from_config, print_params_initial_priors, update_t0_prior_individual, update_t0_prior_folded,
                   run_transit_fitting, supersample_lc_fitted, split_indiviual_lc, plot_trace_evolution, plot_posterior_corner)
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
pytransit_fitting_plots_dir_source_sector_suffix = config["directory"]["pytransit_fitting_plots_dir_source_sector_suffix"]
if pytransit_fitting_plots_dir_source_sector_suffix is not None:
    pytransit_fitting_plots_dir_source_sector += f"{pytransit_fitting_plots_dir_source_sector_suffix}"
os.makedirs(pytransit_fitting_plots_dir_source_sector, exist_ok=True)
pytransit_fitting_plots_suffix = config["directory"]["pytransit_fitting_plots_suffix"] if config["directory"]["pytransit_fitting_plots_suffix"] is not None else ""




# Define the Lightkurve raw-nans-removed light curve BLS-fitted transit parameters
transit_mask_bls_raw_nans_removed_span_coefficient = config['lightkurve']['transit_mask_bls_raw_nans_removed_span_coefficient'] if config['lightkurve']['transit_mask_bls_raw_nans_removed_span_coefficient'] is not None else 1.8 ##### set the coefficient of BLS transit mask span #####
p_bls_raw_nans_removed = config['lightkurve']['raw_nans_removed_bls_fitted_parameters']['p'] if config['lightkurve']['raw_nans_removed_bls_fitted_parameters']['p'] is not None else None
t0_bls_raw_nans_removed = config['lightkurve']['raw_nans_removed_bls_fitted_parameters']['t0'] if config['lightkurve']['raw_nans_removed_bls_fitted_parameters']['t0'] is not None else None
transit_duration_bls_raw_nans_removed = config['lightkurve']['raw_nans_removed_bls_fitted_parameters']['transit_duration'] if config['lightkurve']['raw_nans_removed_bls_fitted_parameters']['transit_duration'] is not None else None
transit_depth_bls_raw_nans_removed = config['lightkurve']['raw_nans_removed_bls_fitted_parameters']['transit_depth'] if config['lightkurve']['raw_nans_removed_bls_fitted_parameters']['transit_depth'] is not None else None




### ------ Lightcurve Selecting ------ ###
# Set the lightcurve selecting and reading parameters
lc_provenance = config['transit_fitting']['lightcurve_provenance']
flux_column = config['transit_fitting']['flux_column']

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
    lc = lk.read(lc_path, flux_column=flux_column)
    lc = lc.normalize() # Force lightcurve normalization before fitting
    print(f"Successfully found and read the specified lightcurve file: {lc_path}.\n")




# Define the lightcurve-specified directory
lc_fn_pure = os.path.splitext(lc_fn)[0]
lc_fn_suffix = lc_fn_pure.replace(f"{name}_{mission}_Sector-{sector}_", "", 1).replace(f"_LC", "", -1)
correction = config['lightkurve']['correction'] if config['lightkurve']['correction'] is not None else ""

pytransit_fitting_plots_dir_source_sector_lc = pytransit_fitting_plots_dir_source_sector + f"/{lc_fn_suffix}"
pytransit_fitting_plots_dir_source_sector_lc_suffix = config["directory"]["pytransit_fitting_plots_dir_source_sector_lc_suffix"]
if pytransit_fitting_plots_dir_source_sector_lc_suffix is not None:
    pytransit_fitting_plots_dir_source_sector_lc += f"{pytransit_fitting_plots_dir_source_sector_lc_suffix}"
os.makedirs(pytransit_fitting_plots_dir_source_sector_lc, exist_ok=True)


# Define the lightcurve plot title
lc_plot_title = lc_fn_pure.replace(f"_{mission}", "", 1).replace(f"{correction}", "").replace("_LC", "", -1).replace("_", " ").replace("Sector-", "Sector ").replace("lightkurve aperture", "lightkurve_aperture")




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
chain_thin_global = config['transit_fitting']['chain_thin_global'] # thinning factor of the MCMC chain for global fitting

# plotting parameters
plot_errorbar = config['transit_fitting']['plot_errorbar'] # whether to plot the error bars of the light curve and residuals
alpha_exptime = config['transit_fitting']['alpha_exptime'] if config['transit_fitting']['alpha_exptime'] is not None else alpha_exptime_default(exptime) # the plotting alpha coefficient of the light curve corresponding to the exposure time
scatter_point_size_exptime = config['transit_fitting']['scatter_point_size_exptime'] if config['transit_fitting']['scatter_point_size_exptime'] is not None else scatter_point_size_exptime_default(exptime) # the scatter point size coefficient of the light curve corresponding to the exposure time

supersample_lcft = config['transit_fitting']['supersample_lc_fitted'] # whether to supersample the best fitted model lightcurve for plotting
supersample_lcft_factor = config['transit_fitting']['supersample_lc_fitted_factor'] # the supersampling factor when supersampling the best fitted model lightcurve for plotting
running_mean_window_proportion_global = config['transit_fitting']['running_mean_window_proportion_global'] # window length proportion of the unflattened MCMC chain to calculate the running means of the parameters when visualizing the process and result for global fitting
running_mean_window_length_global = int(running_mean_window_proportion_global * n_steps_global * (1 - chain_discard_proportion_global) / chain_thin_global) if running_mean_window_proportion_global is not None else None


# Set the global fitted transit parameters names
params_global_name = PARAMS_NAME

# Set the initial global fitted transit parameters and priors
params_global_initial = load_params_initial_from_config(config, 'global', lc)
priors_global = load_priors_from_config(config, 'global', lc)

print_params_initial_priors(params_global_initial, priors_global, 'global', lc)


if fit_global:
    # Run global transit fitting
    fitting_results_global = run_transit_fitting(lc, transit_model_name_global, 'global', params_global_name, params_global_initial, priors_global, n_walkers_global, n_steps_global, chain_discard_proportion_global, chain_thin_global, max_iter_global, sigma_global)
    [n_iter_global,
     params_global_name_full,
     n_params_global_free, params_global_name_free, params_global_name_fixed,
     params_global_initial, priors_global,
     params_global_samples_all, params_global_samples_unflattened_all,
     params_global_best_full, params_global_best_lower_error_full, params_global_best_upper_error_full,
     lc, lc_fitted_global, lc_residual_global,
     residual_std_global, chi_square_global, reduced_chi_square_global,
     r_hat_global, ess_bulk_global, ess_tail_global] = fitting_results_global.values()

    params_global_name_fixed_str = ", ".join(params_global_name_fixed) if len(params_global_name_fixed) > 0 else "None"
    params_global_best_all = {key: params_global_best_full[key] for key in params_global_name}

    config = update_config(config_path, {
        'transit_fitting.global_fitted_transit_parameters.n_iterations': n_iter_global,

        'transit_fitting.global_fitted_transit_parameters.residual_std': residual_std_global,
        'transit_fitting.global_fitted_transit_parameters.chi_square': chi_square_global,
        'transit_fitting.global_fitted_transit_parameters.reduced_chi_square': reduced_chi_square_global,
    })

    for key in params_global_name:
        config = update_config(config_path, {
            f'transit_fitting.global_intitial_transit_parameters.{key}': params_global_initial[f'{key}'],
            f'transit_fitting.global_priors.{key}': priors_global[f'{key}']
        })

    for key in params_global_name_fixed:
        config = update_config(config_path, [
            f'transit_fitting.global_fitted_transit_parameters.r_hat.{key}',
            f'transit_fitting.global_fitted_transit_parameters.ess_bulk.{key}',
            f'transit_fitting.global_fitted_transit_parameters.ess_tail.{key}'
        ], delete=True)

    for key in params_global_name_free:
        config = update_config(config_path, {
            f'transit_fitting.global_fitted_transit_parameters.r_hat.{key}': r_hat_global[f'{key}'],
            f'transit_fitting.global_fitted_transit_parameters.ess_bulk.{key}': ess_bulk_global[f'{key}'],
            f'transit_fitting.global_fitted_transit_parameters.ess_tail.{key}': ess_tail_global[f'{key}']
        })

    for key in params_global_name_full:
        config = update_config(config_path, {
            f'transit_fitting.global_fitted_transit_parameters.{key}': [params_global_best_full[f'{key}'],
                                                                   params_global_best_lower_error_full[f'{key}'],
                                                                   params_global_best_upper_error_full[f'{key}']]
        })


    # Plot the visualization plots of the fitting process and result
    # plot the trace and evolution of MCMC parameters
    j = 1 # count the sub-step

    if running_mean_window_length_global is None or running_mean_window_length_global < 1:
        params_global_trace_evolution_plot = plot_trace_evolution(params_global_samples_unflattened_all)
    else:
        params_global_trace_evolution_plot = plot_trace_evolution(params_global_samples_unflattened_all, running_mean_window_length_global)
        params_global_trace_evolution_plot.axes[1].set_title(f"Evolution ({running_mean_window_proportion_global * 100}% Window Running Mean) Of Parameters", fontsize='x-large')
    params_global_trace_evolution_plot.axes[0].set_title("Trace Of Parameters", fontsize='x-large')
    params_global_trace_evolution_plot.suptitle(f"{lc_plot_title} Global Fitted Transit Parameters Trace and Evolution\n(Fixed Parameters: {params_global_name_fixed_str})", fontsize='xx-large')
    params_global_trace_evolution_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Global_Fitted_Transit_Parameters_Trace_and_Evolution{pytransit_fitting_plots_suffix}.png")
    plt.close()


    # plot the MCMC parameters posterior distribution corner plot
    j += 1 # count the sub-step

    params_global_corner_plot = plot_posterior_corner(params_global_samples_all, quantiles=[0.16, 0.5, 0.84], figsize=(20, 25), show_titles=True, title_fmt=".4f")
    params_global_corner_plot.suptitle(f"{lc_plot_title} Global Fitted Transit Parameters Posterior Distribution Corner Plot", fontsize='xx-large', y=1.05)
    params_global_corner_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Global_Fitted_Transit_Parameters_Posterior_Distribution_Corner_Plot{pytransit_fitting_plots_suffix}.png", bbox_inches='tight')
    plt.close()


    # plot the best fitted light curve and residuals
    j += 1 # count the sub-step

    # supersample the best fitted model lightcurve for plotting
    if supersample_lcft:
        if supersample_lcft_factor is not None and supersample_lcft_factor > 1:
            lc_fitted_global, lc_residual_global = supersample_lc_fitted(params_global_best_all, transit_model_name_global, lc, supersample_lcft_factor)

    # plot the best fitted model
    lc_fitted_global_plot, (ax_lc_fitted_global, ax_lc_residual_global) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    if plot_errorbar:
        lc.scatter(ax=ax_lc_fitted_global, label=None, s=scatter_point_size_exptime / 8, alpha=1.0)
        lc.errorbar(ax=ax_lc_fitted_global, label="Original Light Curve", alpha=alpha_exptime)
    else:
        lc.scatter(ax=ax_lc_fitted_global, label="Original Light Curve", s=scatter_point_size_exptime / 4, alpha=1.0)
    lc_fitted_global.plot(ax=ax_lc_fitted_global, c='red', label=f"Best Fitted {transit_model_name_global} Model, chi-square={chi_square_global:.2f}, reduced chi-square={reduced_chi_square_global:.2f}")
    ax_lc_fitted_global.legend(loc='lower right')
    ax_lc_fitted_global.set_ylabel("Flux")
    ax_lc_fitted_global.tick_params(axis='x', labelbottom=True)
    ax_lc_fitted_global.xaxis.set_major_formatter(ScalarFormatter(useOffset=False)) # disable offset scientific notation
    # plot the best fitted model residuals
    if plot_errorbar:
        lc_residual_global.scatter(ax=ax_lc_residual_global, c='green', label=None, s=scatter_point_size_exptime / 8, alpha=1.0)
        lc_residual_global.errorbar(ax=ax_lc_residual_global, c='green', label=f"Best Fitted Model Residuals, residual std={residual_std_global:.6f}", alpha=alpha_exptime)
    else:
        lc_residual_global.scatter(ax=ax_lc_residual_global, c='green', label=f"Best Fitted Model Residuals, residual std={residual_std_global:.6f}", s=scatter_point_size_exptime / 4, alpha=1.0)
    ax_lc_residual_global.legend(loc='upper right')
    ax_lc_residual_global.set_ylabel("Residuals")
    ax_lc_fitted_global.xaxis.set_major_formatter(ScalarFormatter(useOffset=False)) # disable offset scientific notation

    lc_fitted_global_plot.suptitle(f"{lc_plot_title} Global Best Fitted Light Curve and Residuals")
    lc_fitted_global_plot.figure.tight_layout()
    lc_fitted_global_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Global_Best_Fitted_Light_Curve_and_Residuals{pytransit_fitting_plots_suffix}.png")
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
chain_thin_individual = config['transit_fitting']['chain_thin_individual'] # thinning factor of the MCMC chain for individual fitting

individual_transit_check_coefficient = config['transit_fitting']['individual_transit_check_coefficient'] # the coefficient of transit duration span to check if the individual transit light curve contains transit event

# plotting parameters
running_mean_window_proportion_individual = config['transit_fitting']['running_mean_window_proportion_individual'] # window length proportion of the unflattened MCMC chain to calculate the running means of the parameters when visualizing the process and result for individual fitting
running_mean_window_length_individual = int(running_mean_window_proportion_individual * n_steps_individual * (1 - chain_discard_proportion_individual) / chain_thin_individual) if running_mean_window_proportion_individual is not None else None
individual_transit_plot_coefficient = config['transit_fitting']['individual_transit_plot_coefficient'] # the coefficient of the individual transit plot span

all_in_one_individual_transit_plot_col_row = config['transit_fitting']['all_in_one_individual_transit_plot_col_row'] # (columns, rows) for the all-in-one individual transit plot
if None in all_in_one_individual_transit_plot_col_row:
    all_in_one_individual_transit_plot_col_row = None
if all_in_one_individual_transit_plot_col_row is not None:
    all_in_one_individual_transit_plot_col, all_in_one_individual_transit_plot_row = all_in_one_individual_transit_plot_col_row


# Set the individual fitted transit parameters names
params_individual_name = PARAMS_NAME

# Set the initial individual fitted transit parameters and priors
params_individual_initial = params_global_best_all if fit_global else load_params_initial_from_config(config,
                                                                                                      'individual', lc)
p_individual = params_individual_initial['p']
t0_individual_first_transit = params_individual_initial['t0']

priors_individual = load_priors_from_config(config, 'individual', lc)
priors_individual['p'] = f'fixed({p_individual})'  # fix the period for individual transit fitting
t0_prior_first_transit = priors_individual['t0']

print_params_initial_priors(params_individual_initial, priors_individual, 'individual', lc)

if fit_global:
    transit_duration = params_global_best_full['transit_duration']
elif transit_duration_bls_raw_nans_removed is not None:
    transit_duration = transit_duration_bls_raw_nans_removed * transit_mask_bls_raw_nans_removed_span_coefficient
else:
    transit_duration = transit_duration_nasa


if fit_individual:
    # Split the light curve into individual transit light curves and check if they contain transit events
    lc_individual_list, transit_info = split_indiviual_lc(lc, p_individual, t0_individual_first_transit, transit_duration, individual_transit_check_coefficient)
    n_possible_transits = transit_info['n_possible_transits']
    n_valid_transits = transit_info['n_valid_transits']
    no_data_transit_indices = transit_info['no_data_transit_indices']
    valid_transit_indices = transit_info['valid_transit_indices']

    lc_fitted_individual_list = [None] * n_possible_transits
    lc_residual_individual_list = [None] * n_possible_transits
    individual_transit_plot_range_list = [None] * n_possible_transits
    individual_transit_plot_mask_list = [None] * n_possible_transits

    config = update_config(config_path, {
        'transit_fitting.individual_fitted_transit_parameters.n_possible_transits': n_possible_transits,
        'transit_fitting.individual_fitted_transit_parameters.n_valid_transits': n_valid_transits,
        'transit_fitting.individual_fitted_transit_parameters.no_data_transit_indices': no_data_transit_indices,
        'transit_fitting.individual_fitted_transit_parameters.valid_transit_indices': valid_transit_indices,
    })


    for transit_index in range(n_possible_transits):
        if transit_index in valid_transit_indices:
            lc_individual = lc_individual_list[transit_index]
            # update initial value and prior for t0 for each individual transit fitting
            t0_individual_initial = t0_individual_first_transit + p_individual * transit_index
            params_individual_initial = update_dict(params_individual_initial, {'t0': t0_individual_initial})
            t0_prior = update_t0_prior_individual(t0_prior_first_transit, p_individual, transit_index, lc_individual)
            priors_individual = update_dict(priors_individual, {'t0': t0_prior})

            # Run individual transit fitting
            fitting_results_individual = run_transit_fitting(lc_individual, transit_model_name_individual, 'individual', params_individual_name, params_individual_initial, priors_individual, n_walkers_individual, n_steps_individual, chain_discard_proportion_individual, chain_thin_individual, max_iter_individual, sigma_individual, transit_index)
            [n_iter_individual,
             params_individual_name_full,
             n_params_individual_free, params_individual_name_free, params_individual_name_fixed,
             params_individual_initial, priors_individual,
             params_individual_samples_all, params_individual_samples_unflattened_all,
             params_individual_best_full, params_individual_best_lower_error_full, params_individual_best_upper_error_full,
             lc_individual, lc_fitted_individual, lc_residual_individual,
             residual_std_individual, chi_square_individual, reduced_chi_square_individual,
             r_hat_individual, ess_bulk_individual, ess_tail_individual] = fitting_results_individual.values()

            params_individual_name_fixed_str = ", ".join(params_individual_name_fixed) if len(params_individual_name_fixed) > 0 else "None"
            params_individual_best_all = {key: params_individual_best_full[key] for key in params_individual_name}

            # Store the individual light curve, the individual best fitted light curve and residuals into the lists
            lc_individual_list[transit_index] = lc_individual
            lc_fitted_individual_list[transit_index] = lc_fitted_individual
            lc_residual_individual_list[transit_index] = lc_residual_individual

            if transit_index == valid_transit_indices[0]:
                # Update the individual intitial transit parameters and priors only for the first valid transit
                for key in params_individual_name:
                    config = update_config(config_path, {
                        f'transit_fitting.individual_intitial_transit_parameters.{key}': params_individual_initial[f'{key}'],
                        f'transit_fitting.individual_priors.{key}': priors_individual[f'{key}'],
                    })

                # Initialize the individual fitted transit parameters lists
                config = update_config(config_path, {
                    f'transit_fitting.individual_fitted_transit_parameters.n_iterations': [None] * n_possible_transits,

                    f'transit_fitting.individual_fitted_transit_parameters.residual_std': [None] * n_possible_transits,
                    f'transit_fitting.individual_fitted_transit_parameters.chi_square': [None] * n_possible_transits,
                    f'transit_fitting.individual_fitted_transit_parameters.reduced_chi_square': [None] * n_possible_transits,
                })

                for key in params_individual_name_fixed:
                    config = update_config(config_path, [
                        f'transit_fitting.individual_fitted_transit_parameters.r_hat.{key}',
                        f'transit_fitting.individual_fitted_transit_parameters.ess_bulk.{key}',
                        f'transit_fitting.individual_fitted_transit_parameters.ess_tail.{key}'
                    ], delete=True)

                for key in params_individual_name_free:
                    config = update_config(config_path, {
                        f'transit_fitting.individual_fitted_transit_parameters.r_hat.{key}': [None] * n_possible_transits,
                        f'transit_fitting.individual_fitted_transit_parameters.ess_bulk.{key}': [None] * n_possible_transits,
                        f'transit_fitting.individual_fitted_transit_parameters.ess_tail.{key}': [None] * n_possible_transits
                    })

                for key in params_individual_name_full:
                    if key == 'ldc':
                        config = update_config(config_path, {
                            f'transit_fitting.individual_fitted_transit_parameters.{key}': [[[None, None], [None, None], [None, None]] for t in range(n_possible_transits)]
                        })
                    else:
                        config = update_config(config_path, {
                            f'transit_fitting.individual_fitted_transit_parameters.{key}': [[None, None, None] for t in range(n_possible_transits)]
                        })

            config = update_config(config_path, {
                f'transit_fitting.individual_fitted_transit_parameters.n_iterations[{transit_index}]': n_iter_individual,

                f'transit_fitting.individual_fitted_transit_parameters.residual_std[{transit_index}]': residual_std_individual,
                f'transit_fitting.individual_fitted_transit_parameters.chi_square[{transit_index}]': chi_square_individual,
                f'transit_fitting.individual_fitted_transit_parameters.reduced_chi_square[{transit_index}]': reduced_chi_square_individual,
            })

            for key in params_individual_name_free:
                config = update_config(config_path, {
                    f'transit_fitting.individual_fitted_transit_parameters.r_hat.{key}[{transit_index}]': r_hat_individual[f'{key}'],
                    f'transit_fitting.individual_fitted_transit_parameters.ess_bulk.{key}[{transit_index}]': ess_bulk_individual[f'{key}'],
                    f'transit_fitting.individual_fitted_transit_parameters.ess_tail.{key}[{transit_index}]': ess_tail_individual[f'{key}']
                })

            for key in params_individual_name_full:
                config = update_config(config_path, {
                    f'transit_fitting.individual_fitted_transit_parameters.{key}[{transit_index}]': [params_individual_best_full[f'{key}'],
                                                                                                     params_individual_best_lower_error_full[f'{key}'],
                                                                                                     params_individual_best_upper_error_full[f'{key}']]
                })


            # Plot the visualization plots of the fitting process and result
            # define the directories
            pytransit_fitting_plots_dir_source_sector_lc_transit = pytransit_fitting_plots_dir_source_sector_lc + f"/Transit {transit_index:02}"
            os.makedirs(pytransit_fitting_plots_dir_source_sector_lc_transit, exist_ok=True)


            # plot the trace and evolution of MCMC parameters
            j = 1 # count the sub-step

            if running_mean_window_length_individual is None or running_mean_window_length_individual < 1:
                params_individual_trace_evolution_plot = plot_trace_evolution(params_individual_samples_unflattened_all)
            else:
                params_individual_trace_evolution_plot = plot_trace_evolution(params_individual_samples_unflattened_all, running_mean_window_length_individual)
                params_individual_trace_evolution_plot.axes[1].set_title(f"Evolution ({running_mean_window_proportion_individual * 100}% Window Running Mean) Of Parameters", fontsize='x-large')
            params_individual_trace_evolution_plot.axes[0].set_title("Trace Of Parameters", fontsize='x-large')
            params_individual_trace_evolution_plot.suptitle(f"{lc_plot_title} Individual Transit {transit_index:02} Fitted Transit Parameters Trace and Evolution\n(Fixed Parameters: {params_individual_name_fixed_str})", fontsize='xx-large')
            params_individual_trace_evolution_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc_transit + f"/{i:02}-{j:01}-{transit_index:02}_Individual_Transit-{transit_index:02}_Fitted_Transit_Parameters_Trace_and_Evolution{pytransit_fitting_plots_suffix}.png")
            plt.close()


            # plot the MCMC parameters posterior distribution corner plot
            j += 1 # count the sub-step

            params_individual_corner_plot = plot_posterior_corner(params_individual_samples_all, quantiles=[0.16, 0.5, 0.84], figsize=(20, 25), show_titles=True, title_fmt=".4f")
            params_individual_corner_plot.suptitle(f"{lc_plot_title} Individual Transit {transit_index:02} Fitted Transit Parameters Posterior Distribution Corner Plot", fontsize='xx-large', y=1.05)
            params_individual_corner_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc_transit + f"/{i:02}-{j:01}-{transit_index:02}_Individual_Transit-{transit_index:02}_Fitted_Transit_Parameters_Posterior_Distribution_Corner_Plot{pytransit_fitting_plots_suffix}.png", bbox_inches='tight')
            plt.close()


            # plot the best fitted light curve and residuals
            j += 1 # count the sub-step

            # create the individual transit plot mask, store it into the list and apply it to the lightcurves
            individual_transit_plot_range = (params_individual_best_full['t0'] - params_individual_best_full['transit_duration'] / 2 * individual_transit_plot_coefficient, params_individual_best_full['t0'] + params_individual_best_full['transit_duration'] / 2 * individual_transit_plot_coefficient)
            individual_transit_plot_mask = ((lc_individual.time.value >= individual_transit_plot_range[0]) & (lc_individual.time.value < individual_transit_plot_range[1]))
            individual_transit_plot_range_list[transit_index] = individual_transit_plot_range
            individual_transit_plot_mask_list[transit_index] = individual_transit_plot_mask
            lc_individual_masked = lc_individual[individual_transit_plot_mask]
            lc_fitted_individual_masked = lc_fitted_individual[individual_transit_plot_mask]
            lc_residual_individual_masked = lc_residual_individual[individual_transit_plot_mask]

            # supersample the best fitted model lightcurve for plotting
            if supersample_lcft:
                if supersample_lcft_factor is not None and supersample_lcft_factor > 1:
                    lc_fitted_individual_masked, lc_residual_individual_masked = supersample_lc_fitted(params_individual_best_all, transit_model_name_individual, lc_individual_masked, supersample_lcft_factor)

            # plot the best fitted model
            lc_fitted_individual_plot, (ax_lc_fitted_individual, ax_lc_residual_individual) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
            if plot_errorbar:
                lc_individual_masked.scatter(ax=ax_lc_fitted_individual, label=None, s=scatter_point_size_exptime * 5, alpha=1.0)
                lc_individual_masked.errorbar(ax=ax_lc_fitted_individual, label="Original Light Curve", alpha=alpha_exptime * 5 if alpha_exptime <= 1/5 else 1.0)
            else:
                lc_individual_masked.scatter(ax=ax_lc_fitted_individual, label="Original Light Curve", s=scatter_point_size_exptime * 10, alpha=1.0)
            lc_fitted_individual_masked.plot(ax=ax_lc_fitted_individual, c='red', label=f"Best Fitted {transit_model_name_individual} Model, chi-square={chi_square_individual:.2f}, reduced chi-square={reduced_chi_square_individual:.2f}")
            ax_lc_fitted_individual.legend(loc='lower right')
            ax_lc_fitted_individual.set_ylabel("Flux")
            ax_lc_fitted_individual.tick_params(axis='x', labelbottom=True)
            ax_lc_fitted_individual.xaxis.set_major_formatter(ScalarFormatter(useOffset=False)) # disable offset scientific notation
            ax_lc_fitted_individual.set_xlim(individual_transit_plot_range[0], individual_transit_plot_range[1])
            # plot the best fitted model residuals
            if plot_errorbar:
                lc_residual_individual_masked.scatter(ax=ax_lc_residual_individual, c='green', label=None, s=scatter_point_size_exptime * 5, alpha=1.0)
                lc_residual_individual_masked.errorbar(ax=ax_lc_residual_individual, c='green', label=f"Best Fitted Model Residuals, residual std={residual_std_individual:.6f}", alpha=alpha_exptime * 5 if alpha_exptime <= 1/5 else 1.0)
            else:
                lc_residual_individual_masked.scatter(ax=ax_lc_residual_individual, c='green', label=f"Best Fitted Model Residuals, residual std={residual_std_individual:.6f}", s=scatter_point_size_exptime * 10, alpha=1.0)
            ax_lc_residual_individual.legend(loc='upper right')
            ax_lc_residual_individual.set_ylabel("Residuals")
            ax_lc_residual_individual.xaxis.set_major_formatter(ScalarFormatter(useOffset=False)) # disable offset scientific notation
            ax_lc_residual_individual.set_xlim(individual_transit_plot_range[0], individual_transit_plot_range[1])

            lc_fitted_individual_plot.suptitle(f"{lc_plot_title} Individual Transit {transit_index:02} Best Fitted Light Curve and Residuals")
            lc_fitted_individual_plot.figure.tight_layout()
            lc_fitted_individual_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc_transit + f"/{i:02}-{j:01}-{transit_index:02}_Individual_Transit-{transit_index:02}_Best_Fitted_Light_Curve_and_Residuals{pytransit_fitting_plots_suffix}.png")
            plt.close()


    # plot the all-in-one best fitted light curves
    if all_in_one_individual_transit_plot_col_row is None:
        all_in_one_individual_transit_plot_col = 1
        all_in_one_individual_transit_plot_row = n_valid_transits
    all_in_one_individual_transit_plot_figsize = (20 * all_in_one_individual_transit_plot_col, 5 * all_in_one_individual_transit_plot_row)
    n_individual_transit_subplot = all_in_one_individual_transit_plot_col * all_in_one_individual_transit_plot_row

    lc_fitted_individual_all_plot, axes_lc_fitted_individual = plt.subplots(all_in_one_individual_transit_plot_row, all_in_one_individual_transit_plot_col, figsize=all_in_one_individual_transit_plot_figsize)
    axes_lc_fitted_individual = np.array(axes_lc_fitted_individual).flatten()

    for transit_plot_index, transit_index in enumerate(valid_transit_indices):
        individual_transit_plot_range = individual_transit_plot_range_list[transit_index]
        individual_transit_plot_mask = individual_transit_plot_mask_list[transit_index]
        lc_individual_masked = lc_individual_list[transit_index][individual_transit_plot_mask]
        lc_fitted_individual_masked = lc_fitted_individual_list[transit_index][individual_transit_plot_mask]

        col = transit_plot_index // all_in_one_individual_transit_plot_row
        row = transit_plot_index % all_in_one_individual_transit_plot_row
        ax_lc_fitted_individual = axes_lc_fitted_individual[row * all_in_one_individual_transit_plot_col + col]
        if plot_errorbar:
            lc_individual_masked.scatter(ax=ax_lc_fitted_individual, label=None, s=scatter_point_size_exptime * 5, alpha=1.0)
            lc_individual_masked.errorbar(ax=ax_lc_fitted_individual, label="Original Light Curve" if transit_plot_index % all_in_one_individual_transit_plot_row == 0 else None, alpha=alpha_exptime * 5 if alpha_exptime <= 1/5 else 1.0)
        else:
            lc_individual_masked.scatter(ax=ax_lc_fitted_individual, label="Original Light Curve" if transit_plot_index % all_in_one_individual_transit_plot_row == 0 else None, s=scatter_point_size_exptime * 10, alpha=1.0)
        lc_fitted_individual_masked.plot(ax=ax_lc_fitted_individual, c='red', label=f"Best Fitted {transit_model_name_individual} Model, chi-square={config['transit_fitting']['individual_fitted_transit_parameters']['chi_square'][transit_index]:.2f},\n"
                                                                                    f"reduced chi-square={config['transit_fitting']['individual_fitted_transit_parameters']['reduced_chi_square'][transit_index]:.2f}, residual std={config['transit_fitting']['individual_fitted_transit_parameters']['residual_std'][transit_index]:.6f}" if transit_plot_index % all_in_one_individual_transit_plot_row == 0
                                                                                    else f"chi-square={config['transit_fitting']['individual_fitted_transit_parameters']['chi_square'][transit_index]:.2f}, reduced chi-square={config['transit_fitting']['individual_fitted_transit_parameters']['reduced_chi_square'][transit_index]:.2f},\n"
                                                                                    f"residual std={config['transit_fitting']['individual_fitted_transit_parameters']['residual_std'][transit_index]:.6f}")
        ax_lc_fitted_individual.legend(loc='lower right')
        ax_lc_fitted_individual.set_ylabel("Flux")
        ax_lc_fitted_individual.xaxis.set_major_formatter(ScalarFormatter(useOffset=False)) # disable offset scientific notation
        ax_lc_fitted_individual.set_xlim(individual_transit_plot_range[0], individual_transit_plot_range[1])
        ax_lc_fitted_individual.set_title(f"Transit {transit_index:02}", fontsize='x-large')

    # set unused axes invisible
    for unused_index in range(n_valid_transits, n_individual_transit_subplot):
        ax_unused = axes_lc_fitted_individual[unused_index]
        ax_unused.set_visible(False)

    lc_fitted_individual_all_plot.suptitle(f"{lc_plot_title} All-in-one Individual Best Fitted Light Curves", fontsize='xx-large', y=1.00)
    lc_fitted_individual_all_plot.tight_layout()
    lc_fitted_individual_all_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}_All-in-one_Individual_Best_Fitted_Light_Curves{pytransit_fitting_plots_suffix}.png", bbox_inches='tight')
    plt.close()




##### Define the folding and binning parameters #####
fold = config['transit_fitting']['fold']
p_fold = load_params_fold_from_config(config, fit_global, params_global_best_all if fit_global else None)['p']
t0_fold = load_params_fold_from_config(config, fit_global, params_global_best_all if fit_global else None)['t0']

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
        lc_folded.scatter(ax=ax_lc_folded, label=None, s=scatter_point_size_exptime * p_fold / 20, alpha=1.0)
        lc_folded.errorbar(ax=ax_lc_folded, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_folded_cdpp:.2f} ppm", alpha=alpha_exptime * p_fold / 10 if alpha_exptime <= 10/p_fold else 1.0)
    else:
        lc_folded.scatter(ax=ax_lc_folded, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_folded_cdpp:.2f} ppm", s=scatter_point_size_exptime * p_fold / 10, alpha=1.0)
    ax_lc_folded.legend(loc='lower right')
    ax_lc_folded.set_title(f"{lc_plot_title} Period={p_fold:.4f}d Folded Light Curve")
    ax_lc_folded.set_ylabel("Flux")
    lc_folded_plot.figure.tight_layout()
    lc_folded_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}_Period={p_fold:.4f}d_Folded_Light_Curve{pytransit_fitting_plots_suffix}.png")
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
        lc_fnb.scatter(ax=ax_lc_fnb, label=None, s=scatter_point_size_exptime * 2, alpha=1.0)
        lc_fnb.errorbar(ax=ax_lc_fnb, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_fnb_cdpp:.2f} ppm", alpha=alpha_exptime * 2 if alpha_exptime <= 1/2 else 1.0)
    else:
        lc_fnb.scatter(ax=ax_lc_fnb, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_fnb_cdpp:.2f} ppm", s=scatter_point_size_exptime * 4, alpha=1.0)
    ax_lc_fnb.legend(loc='lower right')
    if fold:
        ax_lc_fnb.set_title(f"{lc_plot_title} Time-bin-size={time_bin_size:.1f} Binned\nPeriod={p_fold:.4f}d Folded Light Curve")
        ax_lc_fnb.set_ylabel("Flux")
        lc_fnb_plot.figure.tight_layout()
        lc_fnb_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}_Time-Bin-Size={time_bin_size.value}s_Binned_Period={p_fold:.4f}d_Folded_Light_Curve{pytransit_fitting_plots_suffix}.png")
    else:
        ax_lc_fnb.set_title(f"{lc_plot_title} Time-bin-size={time_bin_size:.1f} Binned Light Curve")
        ax_lc_fnb.set_ylabel("Flux")
        lc_fnb_plot.figure.tight_layout()
        lc_fnb_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}_Time-Bin-Size={time_bin_size.value}s_Binned_Light_Curve{pytransit_fitting_plots_suffix}.png")
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

if fold or bin:
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
chain_thin_fnb = config['transit_fitting']['chain_thin_fnb'] # thinning factor of the MCMC chain for folded-and-binned fitting

# plotting parameters
running_mean_window_proportion_fnb = config['transit_fitting']['running_mean_window_proportion_fnb'] # window length proportion of the unflattened MCMC chain to calculate the running means of the parameters when visualizing the process and result for folded-and-binned fitting
running_mean_window_length_fnb = int(running_mean_window_proportion_fnb * n_steps_fnb * (1 - chain_discard_proportion_fnb) / chain_thin_fnb) if running_mean_window_proportion_fnb is not None else None
folded_transit_plot_coefficient = config['transit_fitting']['folded_transit_plot_coefficient'] # the coefficient of the folded-and-binned transit plot span, only used when 'fold' is set to true (i.e., the lightcurve is folded)


# Set the folded-and-binned fitted transit parameters names
params_fnb_name = PARAMS_NAME

# Set the initial folded-and-binned fitted transit parameters and priors
params_fnb_initial = params_global_best_all if fit_global else load_params_initial_from_config(config, 'fnb', lc_fnb)
if fold:
    params_fnb_initial['t0'] = 0.0  # set initial epoch time to 0.0 for folded transit fitting
p_fnb = params_fnb_initial['p']

priors_fnb = load_priors_from_config(config, 'fnb', lc_fnb)
if fold:
    priors_fnb['p'] = f'fixed({p_fnb})'  # fix the period for folded transit fitting
    priors_fnb['t0'] = update_t0_prior_folded(priors_fnb['t0'], lc_fnb)

print_params_initial_priors(params_fnb_initial, priors_fnb, 'fnb', lc)


if fit_fnb:
    # Run folded-and-binned transit fitting
    fitting_results_fnb = run_transit_fitting(lc_fnb, transit_model_name_fnb, 'folded' if fold else 'global', params_fnb_name, params_fnb_initial, priors_fnb, n_walkers_fnb, n_steps_fnb, chain_discard_proportion_fnb, chain_thin_fnb, max_iter_fnb, sigma_fnb)
    [n_iter_fnb,
     params_fnb_name_full,
     n_params_fnb_free, params_fnb_name_free, params_fnb_name_fixed,
     params_fnb_initial, priors_fnb,
     params_fnb_samples_all, params_fnb_samples_unflattened_all,
     params_fnb_best_full, params_fnb_best_lower_error_full, params_fnb_best_upper_error_full,
     lc_fnb, lc_fitted_fnb, lc_residual_fnb,
     residual_std_fnb, chi_square_fnb, reduced_chi_square_fnb,
     r_hat_fnb, ess_bulk_fnb, ess_tail_fnb] = fitting_results_fnb.values()

    params_fnb_name_fixed_str = ", ".join(params_fnb_name_fixed) if len(params_fnb_name_fixed) > 0 else "None"
    params_fnb_best_all = {key: params_fnb_best_full[key] for key in params_fnb_name}

    config = update_config(config_path, {
        'transit_fitting.fnb_fitted_transit_parameters.n_iterations': n_iter_fnb,

        'transit_fitting.fnb_fitted_transit_parameters.residual_std': residual_std_fnb,
        'transit_fitting.fnb_fitted_transit_parameters.chi_square': chi_square_fnb,
        'transit_fitting.fnb_fitted_transit_parameters.reduced_chi_square': reduced_chi_square_fnb,
    })

    for key in params_fnb_name:
        config = update_config(config_path, {
            f'transit_fitting.fnb_intitial_transit_parameters.{key}': params_fnb_initial[f'{key}'],
            f'transit_fitting.fnb_priors.{key}': priors_fnb[f'{key}']
        })

    for key in params_fnb_name_fixed:
        config = update_config(config_path, [
            f'transit_fitting.fnb_fitted_transit_parameters.r_hat.{key}',
            f'transit_fitting.fnb_fitted_transit_parameters.ess_bulk.{key}',
            f'transit_fitting.fnb_fitted_transit_parameters.ess_tail.{key}'
        ], delete=True)

    for key in params_fnb_name_free:
        config = update_config(config_path, {
            f'transit_fitting.fnb_fitted_transit_parameters.r_hat.{key}': r_hat_fnb[f'{key}'],
            f'transit_fitting.fnb_fitted_transit_parameters.ess_bulk.{key}': ess_bulk_fnb[f'{key}'],
            f'transit_fitting.fnb_fitted_transit_parameters.ess_tail.{key}': ess_tail_fnb[f'{key}']
        })

    for key in params_fnb_name_full:
        config = update_config(config_path, {
            f'transit_fitting.fnb_fitted_transit_parameters.{key}': [params_fnb_best_full[f'{key}'],
                                                                   params_fnb_best_lower_error_full[f'{key}'],
                                                                   params_fnb_best_upper_error_full[f'{key}']],
        })


    # Plot the visualization plots of the fitting process and result
    # plot the trace and evolution of MCMC parameters
    j = 1  # count the sub-step

    if running_mean_window_length_fnb is None or running_mean_window_length_fnb < 1:
        params_fnb_trace_evolution_plot = plot_trace_evolution(params_fnb_samples_unflattened_all)
    else:
        params_fnb_trace_evolution_plot = plot_trace_evolution(params_fnb_samples_unflattened_all, running_mean_window_length_fnb)
        params_fnb_trace_evolution_plot.axes[1].set_title(f"Evolution ({running_mean_window_proportion_fnb * 100}% Window Running Mean) Of Parameters", fontsize='x-large')
    params_fnb_trace_evolution_plot.axes[0].set_title("Trace Of Parameters", fontsize='x-large')
    if fold and bin:
        params_fnb_trace_evolution_plot.suptitle(f"{lc_plot_title} Folded-and-binned Fitted Transit Parameters Trace and Evolution\n(Fixed Parameters: {params_fnb_name_fixed_str})", fontsize='xx-large')
        params_fnb_trace_evolution_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Folded-and-binned_Fitted_Transit_Parameters_Trace_and_Evolution{pytransit_fitting_plots_suffix}.png")
    elif fold and not bin:
        params_fnb_trace_evolution_plot.suptitle(f"{lc_plot_title} Folded Fitted Transit Parameters Trace and Evolution\n(Fixed Parameters: {params_fnb_name_fixed_str})", fontsize='xx-large')
        params_fnb_trace_evolution_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Folded_Fitted_Transit_Parameters_Trace_and_Evolution{pytransit_fitting_plots_suffix}.png")
    elif bin and not fold:
        params_fnb_trace_evolution_plot.suptitle(f"{lc_plot_title} Binned Fitted Transit Parameters Trace and Evolution\n(Fixed Parameters: {params_fnb_name_fixed_str})", fontsize='xx-large')
        params_fnb_trace_evolution_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Binned_Fitted_Transit_Parameters_Trace_and_Evolution{pytransit_fitting_plots_suffix}.png")
    plt.close()


    # plot the MCMC parameters posterior distribution corner plot
    j += 1  # count the sub-step

    params_fnb_corner_plot = plot_posterior_corner(params_fnb_samples_all, quantiles=[0.16, 0.5, 0.84], figsize=(20, 25), show_titles=True, title_fmt=".4f")
    if fold and bin:
        params_fnb_corner_plot.suptitle(f"{lc_plot_title} Folded-and-binned Fitted Transit Parameters Posterior Distribution Corner Plot", fontsize='xx-large', y=1.05)
        params_fnb_corner_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Folded-and-binned_Fitted_Transit_Parameters_Posterior_Distribution_Corner_Plot{pytransit_fitting_plots_suffix}.png", bbox_inches='tight')
    elif fold and not bin:
        params_fnb_corner_plot.suptitle(f"{lc_plot_title} Folded Fitted Transit Parameters Posterior Distribution Corner Plot", fontsize='xx-large', y=1.05)
        params_fnb_corner_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Folded_Fitted_Transit_Parameters_Posterior_Distribution_Corner_Plot{pytransit_fitting_plots_suffix}.png", bbox_inches='tight')
    elif bin and not fold:
        params_fnb_corner_plot.suptitle(f"{lc_plot_title} Binned Fitted Transit Parameters Posterior Distribution Corner Plot", fontsize='xx-large', y=1.05)
        params_fnb_corner_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Binned_Fitted_Transit_Parameters_Posterior_Distribution_Corner_Plot{pytransit_fitting_plots_suffix}.png", bbox_inches='tight')
    plt.close()


    # plot the best fitted light curve and residuals
    j += 1 # count the sub-step

    # create the folded transit plot mask and apply it to the lightcurve
    if fold:
        folded_transit_plot_range = (params_fnb_best_full['t0'] - params_fnb_best_full['transit_duration'] / 2 * folded_transit_plot_coefficient, params_fnb_best_full['t0'] + params_fnb_best_full['transit_duration'] / 2 * folded_transit_plot_coefficient)
        folded_transit_plot_mask = ((lc_fnb.time.value >= folded_transit_plot_range[0]) & (lc_fnb.time.value < folded_transit_plot_range[1]))
        lc_fnb_masked = lc_fnb[folded_transit_plot_mask]
        lc_fitted_fnb_masked = lc_fitted_fnb[folded_transit_plot_mask]
        lc_residual_fnb_masked = lc_residual_fnb[folded_transit_plot_mask]
    else:
        lc_fnb_masked = lc_fnb
        lc_fitted_fnb_masked = lc_fitted_fnb
        lc_residual_fnb_masked = lc_residual_fnb

    # supersample the best fitted model lightcurve for plotting
    if supersample_lcft:
        if supersample_lcft_factor is not None and supersample_lcft_factor > 1:
            lc_fitted_fnb_masked, lc_residual_fnb_masked = supersample_lc_fitted(params_fnb_best_all, transit_model_name_fnb, lc_fnb_masked, supersample_lcft_factor)

    # plot the best fitted model
    lc_fitted_fnb_plot, (ax_lc_fitted_fnb, ax_lc_residual_fnb) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    if plot_errorbar:
        if fold and bin:
            lc_fnb_masked.scatter(ax=ax_lc_fitted_fnb, label=None, s=scatter_point_size_exptime * 5, alpha=1.0)
            lc_fnb_masked.errorbar(ax=ax_lc_fitted_fnb, label="Original Light Curve", alpha=alpha_exptime * 5 if alpha_exptime <= 1/5 else 1.0)
        elif fold and not bin:
            lc_fnb_masked.scatter(ax=ax_lc_fitted_fnb, label=None, s=scatter_point_size_exptime * p_fold / 10, alpha=1.0)
            lc_fnb_masked.errorbar(ax=ax_lc_fitted_fnb, label="Original Light Curve", alpha=alpha_exptime * p_fold / 5 if alpha_exptime <= 5/p_fold else 1.0)
        elif bin and not fold:
            lc_fnb_masked.scatter(ax=ax_lc_fitted_fnb, label=None, s=scatter_point_size_exptime * 2, alpha=1.0)
            lc_fnb_masked.errorbar(ax=ax_lc_fitted_fnb, label="Original Light Curve", alpha=1.0)
    else:
        if fold and bin:
            lc_fnb_masked.scatter(ax=ax_lc_fitted_fnb, label="Original Light Curve", s=scatter_point_size_exptime * 10, alpha=1.0)
        elif fold and not bin:
            lc_fnb_masked.scatter(ax=ax_lc_fitted_fnb, label="Original Light Curve", s=scatter_point_size_exptime * p_fold / 5, alpha=1.0)
        elif bin and not fold:
            lc_fnb_masked.scatter(ax=ax_lc_fitted_fnb, label="Original Light Curve", s=scatter_point_size_exptime * 4, alpha=1.0)
    lc_fitted_fnb_masked.plot(ax=ax_lc_fitted_fnb, c='red', label=f"Best Fitted {transit_model_name_fnb} Model, chi-square={chi_square_fnb:.2f}, reduced chi-square={reduced_chi_square_fnb:.2f}")
    ax_lc_fitted_fnb.legend(loc='lower right')
    ax_lc_fitted_fnb.set_ylabel("Flux")
    ax_lc_fitted_fnb.tick_params(axis='x', labelbottom=True)
    ax_lc_fitted_fnb.xaxis.set_major_formatter(ScalarFormatter(useOffset=False)) # disable offset scientific notation
    # plot the best fitted model residuals
    if plot_errorbar:
        if fold and bin:
            lc_residual_fnb_masked.scatter(ax=ax_lc_residual_fnb, c='green', label=None, s=scatter_point_size_exptime * 5, alpha=1.0)
            lc_residual_fnb_masked.errorbar(ax=ax_lc_residual_fnb, c='green', label=f"Best Fitted Model Residuals, residual std={residual_std_fnb:.6f}", alpha=alpha_exptime * 5 if alpha_exptime <= 1/5 else 1.0)
        elif fold and not bin:
            lc_residual_fnb_masked.scatter(ax=ax_lc_residual_fnb, c='green', label=None, s=scatter_point_size_exptime * p_fold / 10, alpha=1.0)
            lc_residual_fnb_masked.errorbar(ax=ax_lc_residual_fnb, c='green', label=f"Best Fitted Model Residuals, residual std={residual_std_fnb:.6f}", alpha=alpha_exptime * p_fold / 5 if alpha_exptime <= 5/p_fold else 1.0)
        elif bin and not fold:
            lc_residual_fnb_masked.scatter(ax=ax_lc_residual_fnb, c='green', label=None, s=scatter_point_size_exptime * 2, alpha=1.0)
            lc_residual_fnb_masked.errorbar(ax=ax_lc_residual_fnb, c='green', label=f"Best Fitted Model Residuals, residual std={residual_std_fnb:.6f}", alpha=1.0)
    else:
        if fold and bin:
            lc_residual_fnb_masked.scatter(ax=ax_lc_residual_fnb, c='green', label=f"Best Fitted Model Residuals, residual std={residual_std_fnb:.6f}", s=scatter_point_size_exptime * 10, alpha=1.0)
        elif fold and not bin:
            lc_residual_fnb_masked.scatter(ax=ax_lc_residual_fnb, c='green', label=f"Best Fitted Model Residuals, residual std={residual_std_fnb:.6f}", s=scatter_point_size_exptime * p_fold / 5, alpha=1.0)
        elif bin and not fold:
            lc_residual_fnb_masked.scatter(ax=ax_lc_residual_fnb, c='green', label=f"Best Fitted Model Residuals, residual std={residual_std_fnb:.6f}", s=scatter_point_size_exptime * 4, alpha=1.0)
    ax_lc_residual_fnb.legend(loc='upper right')
    ax_lc_residual_fnb.set_ylabel("Residuals")
    ax_lc_fitted_fnb.xaxis.set_major_formatter(ScalarFormatter(useOffset=False)) # disable offset scientific notation

    if fold and bin:
        lc_fitted_fnb_plot.suptitle(f"{lc_plot_title} Folded-and-binned Best Fitted Light Curve and Residuals")
        lc_fitted_fnb_plot.figure.tight_layout()
        lc_fitted_fnb_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Folded-and-binned_Best_Fitted_Light_Curve_and_Residuals{pytransit_fitting_plots_suffix}.png")
    elif fold and not bin:
        lc_fitted_fnb_plot.suptitle(f"{lc_plot_title} Folded Best Fitted Light Curve and Residuals")
        lc_fitted_fnb_plot.figure.tight_layout()
        lc_fitted_fnb_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Folded_Best_Fitted_Light_Curve_and_Residuals{pytransit_fitting_plots_suffix}.png")
    elif bin and not fold:
        lc_fitted_fnb_plot.suptitle(f"{lc_plot_title} Binned Best Fitted Light Curve and Residuals")
        lc_fitted_fnb_plot.figure.tight_layout()
        lc_fitted_fnb_plot.figure.savefig(pytransit_fitting_plots_dir_source_sector_lc + f"/{i:02}-{j:01}_Binned_Best_Fitted_Light_Curve_and_Residuals{pytransit_fitting_plots_suffix}.png")
    plt.close()