from tqdm import tqdm
import os
import time

import lightkurve as lk
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import rcParams

from utils import format_fits_fn, aperture_overlay, calculate_cdpp
from A_ab_Configuration_Loader import *




# Define the directories
tpf_downloaded_dir = data_dir + config['directory']['tpf_downloaded_dir']
os.makedirs(tpf_downloaded_dir, exist_ok=True)
tpf_downloaded_dir_source = tpf_downloaded_dir + f"/{name}"
os.makedirs(tpf_downloaded_dir_source, exist_ok=True)

lc_extracted_dir = data_dir + config['directory']['lc_extracted_dir']
os.makedirs(lc_extracted_dir, exist_ok=True)
lc_extracted_dir_source = lc_extracted_dir + f"/{name}"
os.makedirs(lc_extracted_dir_source, exist_ok=True)

tpf_extracted_lightcurve_plots_dir = base_dir + config['directory']['tpf_extracted_lightcurve_plots_dir']
os.makedirs(tpf_extracted_lightcurve_plots_dir, exist_ok=True)
tpf_extracted_lightcurve_plots_dir_source_sector = tpf_extracted_lightcurve_plots_dir + f"/{name}_Sector-{sector}"
os.makedirs(tpf_extracted_lightcurve_plots_dir_source_sector, exist_ok=True)




### ------ TPF Selecting ------ ###
# Set the TPF selecting criteria
tpf_author = config['tpf_selecting_criteria']['author']
exptime = config['tpf_selecting_criteria']['exptime'] if config['tpf_selecting_criteria']['exptime'] is not None else exptime
tpf_height = config['tpf_selecting_criteria']['tpf_height']
tpf_width = config['tpf_selecting_criteria']['tpf_width']

tpf_selected_fits_metadata_dict = {'type': 'tpf', 'name': name, 'mission': mission, 'sector': sector, 'author': tpf_author, 'exptime': exptime, 'tpf_height': tpf_height, 'tpf_width': tpf_width}
tpf_selected_fn = f"/{format_fits_fn(tpf_selected_fits_metadata_dict)}"
tpf_selected_path = tpf_downloaded_dir_source + tpf_selected_fn


# Read the TPF file via Lightkurve
if os.path.exists(tpf_selected_path):
    tpf_selected = lk.read(tpf_selected_path)
else:
    raise FileNotFoundError(f"TPF file not found in the data directory: {tpf_selected_path} with the specified criteria: name: {name}, mission: {mission}, sector: {sector}, author: {tpf_author}, exptime={exptime}s, tpf_height: {tpf_height}, tpf_width: {tpf_width}.\n"
                            f"Please run B_aab_TPF_Downloading to download the corresponding TPF (or move it to the data directory if it already exists in the cache directory) first.")




### ------ Light Curve Extracting ------ ###
# Set the light curve extracting method
lightcurve_extracting_method = config['lightcurve_extracting']['method']

# Define the author
author = f"{tpf_author}-{lightcurve_extracting_method}-Extracted" ##### set the author of the data to be processed #####


# Define the directories
tpf_extracted_lightcurve_plots_dir_source_sector_tpf_exptime = tpf_extracted_lightcurve_plots_dir_source_sector + f"/Author-{tpf_author}_{tpf_width}x{tpf_height}_Exptime={exptime}s"
os.makedirs(tpf_extracted_lightcurve_plots_dir_source_sector_tpf_exptime, exist_ok=True)
tpf_extracted_lightcurve_plots_dir_source_sector_tpf_exptime_method = tpf_extracted_lightcurve_plots_dir_source_sector_tpf_exptime + f"/{lightcurve_extracting_method}"
os.makedirs(tpf_extracted_lightcurve_plots_dir_source_sector_tpf_exptime_method, exist_ok=True)





### lightkurve_aperture ###
if lightcurve_extracting_method == 'lightkurve_aperture':
    aperture_mask_type = config['lightcurve_extracting']['lightkurve_aperture']['aperture_mask_type']
    threshold = config['lightcurve_extracting']['lightkurve_aperture']['threshold']
    flux_method = config['lightcurve_extracting']['lightkurve_aperture']['flux_method']
    centroid_method = config['lightcurve_extracting']['lightkurve_aperture']['centroid_method']


    # Extract the light curve from the selected TPF
    if aperture_mask_type == 'custom':
        aperture_mask = np.zeros((tpf_selected.flux.shape[1], tpf_selected.flux.shape[2]), dtype=float) ##### create a custom aperture mask #####
        lc_extracted = tpf_selected.extract_aperture_photometry(aperture_mask=aperture_mask, flux_method=flux_method, centroid_method=centroid_method)
        aperture_mask_type = "custom-zeros" ##### rename the aperture mask type #####
        config = update_config(config_path, {'lightcurve_extracting.lightkurve_aperture.aperture_mask_type': f"{aperture_mask_type}"})
    elif aperture_mask_type == 'pipeline':
        if tpf_author.lower() == 'tesscut':
            raise ValueError("The 'pipeline' aperture mask is not available when the TPF author is 'tesscut', please choose another aperture mask type and update the 'aperture_mask_type' parameter accordingly in the configuration file.")
        lc_extracted = tpf_selected.extract_aperture_photometry(aperture_mask=aperture_mask_type, flux_method=flux_method, centroid_method=centroid_method)
        aperture_mask = tpf_selected.pipeline_mask.astype(float)
    elif aperture_mask_type == 'default':
        if tpf_author.lower() == 'tesscut':
            aperture_mask = tpf_selected.create_threshold_mask(threshold=3.0).astype(float) # a 3.0-sigma threshold mask will be the fallback if no pipeline mask is available
            lc_extracted = tpf_selected.extract_aperture_photometry(aperture_mask=aperture_mask, flux_method=flux_method, centroid_method=centroid_method)
            aperture_mask_type = "threshold-3.0-sigma" ##### rename the aperture mask type #####
            config = update_config(config_path, {'lightcurve_extracting.lightkurve_aperture.aperture_mask_type': f"{aperture_mask_type}"})
        else:
            lc_extracted = tpf_selected.extract_aperture_photometry(aperture_mask=aperture_mask_type, flux_method=flux_method, centroid_method=centroid_method)
            if tpf_selected.pipeline_mask is not None:
                aperture_mask = tpf_selected.pipeline_mask.astype(float)
                aperture_mask_type = "pipeline" ##### rename the aperture mask type #####
                config = update_config(config_path, {'lightcurve_extracting.lightkurve_aperture.aperture_mask_type': f"{aperture_mask_type}"})
            else:
                aperture_mask = tpf_selected.create_threshold_mask(threshold=3.0).astype(float) # a 3.0-sigma threshold mask will be the fallback if no pipeline mask is available
                aperture_mask_type = "threshold-3.0-sigma" ##### rename the aperture mask type #####
                config = update_config(config_path, {'lightcurve_extracting.lightkurve_aperture.aperture_mask_type': f"{aperture_mask_type}"})
    elif aperture_mask_type == 'threshold':
        aperture_mask = tpf_selected.create_threshold_mask(threshold=threshold).astype(float)
        lc_extracted = tpf_selected.extract_aperture_photometry(aperture_mask=aperture_mask, flux_method=flux_method, centroid_method=centroid_method)
        aperture_mask_type = f"threshold-{threshold}-sigma" ##### rename the aperture mask type #####
        config = update_config(config_path, {'lightcurve_extracting.lightkurve_aperture.aperture_mask_type': f"{aperture_mask_type}"})
    elif aperture_mask_type == 'all' or aperture_mask_type is None:
        lc_extracted = tpf_selected.extract_aperture_photometry(aperture_mask=aperture_mask_type, flux_method=flux_method, centroid_method=centroid_method)
        aperture_mask = np.ones((tpf_selected.flux.shape[1], tpf_selected.flux.shape[2]), dtype=float)
        aperture_mask_type = "all" ##### rename the aperture mask type #####
        config = update_config(config_path, {'lightcurve_extracting.lightkurve_aperture.aperture_mask_type': f"{aperture_mask_type}"})

    # Save the extracted light curve into a FITS file
    lc_extracted_fits_metadata_dict = {'type':'lc', 'name':name, 'mission':mission, 'sector':sector, 'author':author, 'exptime':exptime, 'tpf_height': tpf_height, 'tpf_width': tpf_width, 'aperture_mask_type': aperture_mask_type, 'flux_method': flux_method, 'centroid_method': centroid_method}
    lc_extracted_fn = f"/{format_fits_fn(lc_extracted_fits_metadata_dict)}"
    lc_extracted_path = lc_extracted_dir_source + lc_extracted_fn
    lc_extracted.to_fits(path=lc_extracted_path, overwrite=True)

    # Calculate the CDPP of the extracted light curve and update the configurations
    lc_extracted_cdpp = calculate_cdpp(lc_extracted, exptime=exptime, cdpp_transit_duration=cdpp_transit_duration)
    config = update_config(config_path, {'lightcurve_extracting.lc_extracted_cdpp': lc_extracted_cdpp})

    print(f"Successfully extracted the light curve from the selected TPF and saved it to the data directory of the source: {lc_extracted_path}.\n")


    # Render the aperture-overlay TPF animation
    i = 1 # count the step

    render_aperture_overlay_tpf_animation = config['lightcurve_extracting']['lightkurve_aperture']['render_aperture_overlay_tpf_animation']
    aperture_overlay_tpf_animation_step = config['lightcurve_extracting']['lightkurve_aperture']['aperture_overlay_tpf_animation_step']

    if render_aperture_overlay_tpf_animation:
        tpf_animation_start_time = time.time()  # measure the start time
        print(f"Rendering the {name} Sector {sector} aperture-overlay TPF animation...")
        tpf_animation_framerate = 25 # set the framerate
        tpf_animation_plot, ax_tpf_animation = plt.subplots(figsize=(10, 10))
        tpf_animation_artists = []
        for k in tqdm(range(0, len(tpf_selected), aperture_overlay_tpf_animation_step), desc="Rendering frames of aperture-overlay TPF animation", unit=" frame"):
            aperture_overlay(tpf_selected.flux, aperture_mask=aperture_mask, data_type='Flux', cadence=k, ax=ax_tpf_animation, show_colorbar=False)
            ax_tpf_animation_cadence_img = ax_tpf_animation.get_images()[-1] # get the newest plotted aperture-overlay TPF
            ax_tpf_animation.set_title("") # clear the default title
            ax_tpf_animation_cadence_title = ax_tpf_animation.text(0.5, 1.01, f"{name} Sector {sector} Aperture {aperture_mask_type} TPF (Cadence {k:04})", ha='center', transform=ax_tpf_animation.transAxes, fontfamily=rcParams['font.family'], fontsize='xx-large', fontweight=rcParams['axes.titleweight'])
            tpf_animation_artists.append([ax_tpf_animation_cadence_img, ax_tpf_animation_cadence_title])
        print("Frames rendered, now compiling the animation...")
        tpf_animation = animation.ArtistAnimation(fig=tpf_animation_plot, artists=tpf_animation_artists, interval=int(1000/tpf_animation_framerate), blit=True)
        tpf_animation_writer = animation.FFMpegWriter(fps=int(tpf_animation_framerate), metadata=dict(artist='TELCEF'), bitrate=2000)
        tpf_animation.save(filename=tpf_extracted_lightcurve_plots_dir_source_sector_tpf_exptime_method + f"/{i:02}_{name}_Sector-{sector}_Aperture-{aperture_mask_type}_Step-{aperture_overlay_tpf_animation_step}_TPF_Animation.mp4", writer=tpf_animation_writer)
        tpf_animation_end_time = time.time()  # measure the end time
        tpf_animation_rendering_time = tpf_animation_end_time - tpf_animation_start_time # calculate the rendering time
        print(f"Rendered {name} Sector {sector} Aperture {aperture_mask_type} TPF animation in {tpf_animation_rendering_time:.3f} seconds.\n")
        config = update_config(config_path, {'lightcurve_extracting.lightkurve_aperture.tpf_animation_rendering_time': round(tpf_animation_rendering_time, 3)})
    else:
        print(f"Skipped {name} Sector {sector} aperture-overlay TPF animation rendering.\n")


    # Plot the aperture-overlay TPF and the extracted light curve
    i += 1 # count the step

    k = config['lightcurve_extracting']['lightkurve_aperture']['tpf_plot_cadence'] # set the cadence index
    ax_lc_ylim = config['lightcurve_extracting']['lightkurve_aperture']['lc_plot_ylim'] # set the y-axis limit of the light curve plot
    plot_errorbar = config['lightcurve_extracting']['lightkurve_aperture']['plot_errorbar'] # set whether to plot the error bar of the light curve

    tpf_lc_plot, (ax_tpf, ax_lc) = plt.subplots(1, 2, figsize=(25, 5), gridspec_kw={'width_ratios': [1, 3]})
    aperture_overlay(tpf_selected.flux, aperture_mask=aperture_mask, data_type='Flux', cadence=k, ax=ax_tpf, show_colorbar=True)
    ax_tpf.set_title(f"{name} Sector {sector} Aperture {aperture_mask_type}\nTPF (Cadence {k:04}) Exptime={exptime}s")
    if plot_errorbar:
        lc_extracted.normalize().scatter(ax=ax_lc, label=None, s=0.1)
        lc_extracted.normalize().errorbar(ax=ax_lc, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_extracted_cdpp:.2f} ppm")
    else:
        lc_extracted.normalize().scatter(ax=ax_lc, label=f"{cdpp_transit_duration:.3f}h-CDPP={lc_extracted_cdpp:.2f} ppm", s=0.1)
    if ax_lc_ylim is not None and all(ylim is not None for ylim in ax_lc_ylim):
        ax_lc.set_ylim(np.percentile(lc_extracted.remove_nans().normalize().flux, ax_lc_ylim[0]), np.percentile(lc_extracted.remove_nans().normalize().flux, ax_lc_ylim[1]))
        ax_lc.set_title(f"{name} Sector {sector} Aperture {aperture_mask_type} Light Curve ({ax_lc_ylim[0]}% - {ax_lc_ylim[1]}%) Exptime={exptime}s")
    else:
        ax_lc.set_title(f"{name} Sector {sector} Aperture {aperture_mask_type} Light Curve Exptime={exptime}s")
    tpf_lc_plot.tight_layout(pad=2.16)
    tpf_lc_plot.figure.savefig(tpf_extracted_lightcurve_plots_dir_source_sector_tpf_exptime_method + f"/{i:02}_{name}_Sector-{sector}_Aperture-{aperture_mask_type}_TPF_LC.png")
    plt.close()