from tqdm import tqdm
import shutil
import os
import warnings
import time

import eleanor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import rcParams

from utils import update_config, format_fits_fn, update_fits_headers, calculate_cdpp
from A_ab_Configuration_Loader import *




# Define the directories
eleanor_lc_dir = data_dir + config['directory']['eleanor_lc_dir']
os.makedirs(eleanor_lc_dir, exist_ok=True)
eleanor_lc_dir_source = eleanor_lc_dir + f"/{name}"
os.makedirs(eleanor_lc_dir_source, exist_ok=True)

eleanor_processed_lightcurve_plots_dir = base_dir + config['directory']['eleanor_processed_lightcurve_plots_dir']
os.makedirs(eleanor_processed_lightcurve_plots_dir, exist_ok=True)
eleanor_processed_lightcurve_plots_dir_source_sector = eleanor_processed_lightcurve_plots_dir + f"/{name}_Sector-{sector}"
os.makedirs(eleanor_processed_lightcurve_plots_dir_source_sector, exist_ok=True)


# Define the author
author = "Eleanor" ##### set the author of the data to be processed #####




### ------ Source ------ ###
# Set whether to use postcard or tesscut
postcard = config['eleanor']['source']['postcard'] ##### set whether to use postcard or not #####

# Validate the 'tesscut' parameter
if config['eleanor']['source']['tesscut'] is not postcard:
    tesscut = config['eleanor']['source']['tesscut']
else:
    warnings.warn("The 'tesscut' parameter in the configuration file is set to the same as the 'postcard' parameter. It should be and has been automatically set to the opposite of 'postcard'.")
    tesscut = not postcard
    config = update_config(config_path, {'eleanor.source.tesscut': tesscut})
tesscut_size_1d = config['eleanor']['source']['tesscut_size_1d'] ##### set the TESScut size #####


# Retrieve the source
if postcard:
    post_dir = eleanor_root
elif tesscut:
    post_dir = eleanor_root_tesscut
source = eleanor.Source(name=name, sector=sector, tc=tesscut, tesscut_size=tesscut_size_1d, post_dir=post_dir)


# Update the configurations and move the downloaded tesscut file to eleanor_root_tesscut if 'postcard' is set to 'true' but no postcard is found
if postcard and source.tc == True:
    warnings.warn(f"The 'postcard' parameter in the configuration file is set to 'true', but no postcard is found for {name} in Sector {sector}.\n"
                  f"The 'postcard' parameter has been automatically set to 'false', and the 'tesscut' parameter has been automatically set to 'true'.")

    config = update_config(config_path, {'eleanor.source.postcard': False, 'eleanor.source.tesscut': True})

    for fn in os.listdir(eleanor_root):
        path_ori = os.path.join(eleanor_root, fn)
        if (
            os.path.isfile(path_ori)
            and fn.endswith('.fits')
            and 'astrocut' in fn
        ):
            path_new = os.path.join(eleanor_root_tesscut, fn)
            if os.path.exists(path_new):
                os.remove(path_new)
            shutil.move(path_ori, path_new)


# Reload the configurations
config = load_config(config_path)

postcard = config['eleanor']['source']['postcard']
postcard_size = (148, 104) if postcard else None

tesscut = config['eleanor']['source']['tesscut']
tesscut_size = (tesscut_size_1d, tesscut_size_1d) if tesscut else None


# Define the postcard-specified directories based on whether postcard or tesscut is used
if postcard:
    eleanor_processed_lightcurve_plots_dir_source_sector_pc = eleanor_processed_lightcurve_plots_dir_source_sector + f"/Postcard"
elif tesscut:
    eleanor_processed_lightcurve_plots_dir_source_sector_pc = eleanor_processed_lightcurve_plots_dir_source_sector + f"/TESScut"
os.makedirs(eleanor_processed_lightcurve_plots_dir_source_sector_pc, exist_ok=True)




### ------ TargetData ------ ###
# Set the parameters for TargetData
do_pca = config['eleanor']['targetdata']['do_pca']
do_psf = config['eleanor']['targetdata']['do_psf']
tpf_height = config['eleanor']['targetdata']['tpf_height']
tpf_width = config['eleanor']['targetdata']['tpf_width']
bkg_size = config['eleanor']['targetdata']['bkg_size']
aperture_mode = config['eleanor']['targetdata']['aperture_mode']
cal_cadences = config['eleanor']['targetdata']['cal_cadences']
regressors = config['eleanor']['targetdata']['regressors']


# Retrieve the target data
data = eleanor.TargetData(source, height=tpf_height, width=tpf_width, bkg_size=bkg_size, aperture_mode=aperture_mode, cal_cadences=cal_cadences, do_pca=do_pca, do_psf=do_psf, regressors=regressors)
eleanor_targetdata_fits_metadata_dict = {'type':'targetdata', 'name':name, 'mission':mission, 'sector':sector, 'author':author, 'exptime':exptime, 'postcard':postcard}
data.save(output_fn=f"{format_fits_fn(eleanor_targetdata_fits_metadata_dict)}", directory=eleanor_lc_dir_source)

# Retrieve the specific metadata of the source and update & reload the configurations
config = update_config(config_path, {
    'eleanor.source.location_on_postcard/tesscut': (float(data.cen_x), float(data.cen_y)) if postcard else (float(np.floor((tesscut_size_1d + 1) / 2)), float(np.floor((tesscut_size_1d + 1) / 2))),
    'eleanor.targetdata.location_on_tpf': (float(data.tpf_star_x), float(data.tpf_star_y)),
    'eleanor.targetdata.bkg_type': str(data.bkg_type)
})


# Convert the targetdata to lightkurve.lightcurve.LightCurve() objects and save them into FITS files
# use eleanor.TargetData.quality if quality_mask is set to 'default', otherwise use the custom quality_mask
if config['eleanor']['targetdata']['quality_mask'].lower() == 'default':
    lc_raw = data.to_lightkurve(flux=data.raw_flux, quality_mask=data.quality)
    lc_corr = data.to_lightkurve(flux=data.corr_flux, quality_mask=data.quality)
    lc_pca = data.to_lightkurve(flux=data.pca_flux, quality_mask=data.quality)
    lc_psf = data.to_lightkurve(flux=data.psf_flux, quality_mask=data.quality)
elif isinstance(config['eleanor']['targetdata']['quality_mask'], np.ndarray):
    quality_mask = config['eleanor']['targetdata']['quality_mask']
    lc_raw = data.to_lightkurve(flux=data.raw_flux, quality_mask=quality_mask)
    lc_corr = data.to_lightkurve(flux=data.corr_flux, quality_mask=quality_mask)
    lc_pca = data.to_lightkurve(flux=data.pca_flux, quality_mask=quality_mask)
    lc_psf = data.to_lightkurve(flux=data.psf_flux, quality_mask=quality_mask)
else:
    raise ValueError("The 'quality_mask' parameter in the configuration file should be set to 'default' or a numpy ndarray.")

lc_list = [lc_raw, lc_corr, lc_pca, lc_psf]
lc_type_list = ['Raw', 'Corrected' , 'PCA', 'PSF']
eleanor_lc_fn_list = []
eleanor_lc_path_list = []
for lc_type in lc_type_list:
    eleanor_lc_fits_metadata_dict = {'type':'lc', 'name': name, 'mission': mission, 'sector' :sector, 'author': author, 'exptime': exptime, 'lc_type': lc_type, 'postcard': postcard}
    eleanor_lc_fn = f"/{format_fits_fn(eleanor_lc_fits_metadata_dict)}"
    eleanor_lc_fn_list.append(eleanor_lc_fn)
    eleanor_lc_path = eleanor_lc_dir_source + eleanor_lc_fn
    eleanor_lc_path_list.append(eleanor_lc_path)
for k in range(len(lc_list)):
    lc_list[k].to_fits(path=eleanor_lc_path_list[k], overwrite=True)
    eleanor_lc_headers_update_dict = {0: {'TICID': (tic, "unique tess target identifier"), 'KEPLERID': None,
                                          'MISSION': 'TESS', 'TELESCOP': 'TESS', 'INSTRUME': 'TESS Photometer',
                                          'SECTOR': (sector, "Observing sector"), 'CAMERA': (camera, "Camera number"), 'CCD': (ccd, "CCD chip number"),
                                          'ORIGIN': 'Eleanor', 'CREATOR': f'Eleanor-{lc_type_list[k]}',
                                          'EXPTIME': (exptime, "exposure time in seconds")}}
    update_fits_headers(eleanor_lc_path_list[k], eleanor_lc_headers_update_dict)
    print(f"Successfully extracted the {lc_type_list[k]} light curve of the source via Eleanor and saved it to the data directory of the source: {eleanor_lc_path_list[k]}.")
print("")


# Calculate the CDPP of the light curves, select the type of light curve to be processed based on CDPP and update the configurations
lc_raw_cdpp = calculate_cdpp(lc_raw, exptime=exptime, cdpp_transit_duration=cdpp_transit_duration)
lc_corr_cdpp = calculate_cdpp(lc_corr, exptime=exptime, cdpp_transit_duration=cdpp_transit_duration)
lc_pca_cdpp = calculate_cdpp(lc_pca, exptime=exptime, cdpp_transit_duration=cdpp_transit_duration)
lc_psf_cdpp = calculate_cdpp(lc_psf, exptime=exptime, cdpp_transit_duration=cdpp_transit_duration)
lc_cdpp_list = [lc_raw_cdpp, lc_corr_cdpp, lc_pca_cdpp, lc_psf_cdpp]
config = update_config(config_path, {'eleanor.lc_raw_cdpp': lc_raw_cdpp, 'eleanor.lc_corr_cdpp': lc_corr_cdpp, 'eleanor.lc_pca_cdpp': lc_pca_cdpp, 'eleanor.lc_psf_cdpp': lc_psf_cdpp})

lc_type = lc_type_list[np.argmin(lc_cdpp_list)]
config = update_config(config_path, {'eleanor.lc_type': lc_type})




### ------ Visualize ------ ###
# Create a Visualize object
visualize = eleanor.Visualize(data)




### TPF ###
# Render the TPF animation
i = 1 # count the step

# Set whether to render the (aperture-overplotted) TPF animation
render_tpf_animation = config['eleanor']['visualize']['render_tpf_animation']
render_aperture_overplotted_tpf_animation = config['eleanor']['visualize']['render_aperture_overplotted_tpf_animation']
# validate the 'render_aperture_overplotted_tpf_animation' parameter
if not render_tpf_animation and render_aperture_overplotted_tpf_animation:
    warnings.warn("The 'render_aperture_overplotted_tpf_animation' parameter in the configuration file is set to 'true',\n"
                  "but the 'render_tpf_animation' parameter is set to 'false'. The 'render_aperture_overplotted_tpf_animation' parameter has been automatically set to 'false'.")
    render_aperture_overplotted_tpf_animation = False
    config = update_config(config_path, {'eleanor.visualize.render_aperture_overplotted_tpf_animation': False})

config = load_config(config_path)
render_tpf_animation = config['eleanor']['visualize']['render_tpf_animation'] ##### set whether to render the TPF animation #####
render_aperture_overplotted_tpf_animation = config['eleanor']['visualize']['render_aperture_overplotted_tpf_animation'] ##### set whether to render the aperture-overplotted TPF animation #####
tpf_animation_step = config['eleanor']['visualize']['tpf_animation_step'] ##### set the step size of the TPF animation #####


if render_tpf_animation:
    tpf_animation_start_time = time.time()  # measure the start time
    if render_aperture_overplotted_tpf_animation:
        print(f"Rendering the {name} Sector {sector} Eleanor TPF animation WITH aperture overplotted...")
    else:
        print(f"Rendering the {name} Sector {sector} Eleanor TPF animation WITHOUT aperture overplotted...")
    tpf_animation_framerate = 25 # set the framerate
    tpf_animation_plot, ax_tpf_animation = plt.subplots(figsize=(10, 10))
    tpf_animation_artists = []
    for k in tqdm(range(0, len(data.tpf), tpf_animation_step), desc="Rendering frames of TPF animation", unit=" frame"):
        if render_aperture_overplotted_tpf_animation:
            visualize.aperture_contour(ax=ax_tpf_animation, cadence=k, aperture=data.aperture)
            ax_tpf_animation_cadence_img = ax_tpf_animation.get_images()[-1] # get the newest plotted aperture-overplotted TPF
        else:
            ax_tpf_animation_cadence_img = ax_tpf_animation.imshow(data.tpf[k], origin='lower', animated=True)
        ax_tpf_animation_cadence_title = ax_tpf_animation.text(0.5, 1.01, f"{name} Sector {sector} Eleanor TPF (Cadence {k:04}) Exptime={exptime}s", ha='center', transform=ax_tpf_animation.transAxes, fontfamily=rcParams['font.family'], fontsize='xx-large', fontweight=rcParams['axes.titleweight'])
        tpf_animation_artists.append([ax_tpf_animation_cadence_img, ax_tpf_animation_cadence_title])
    print("Frames rendered, now compiling the animation...")
    tpf_animation = animation.ArtistAnimation(fig=tpf_animation_plot, artists=tpf_animation_artists, interval=int(1000/tpf_animation_framerate), blit=True)
    tpf_animation_writer = animation.FFMpegWriter(fps=int(tpf_animation_framerate), metadata=dict(artist='TELCEF'), bitrate=2000)
    tpf_animation.save(filename=eleanor_processed_lightcurve_plots_dir_source_sector_pc + f"/{i:02}_{name}_Sector-{sector}_Eleanor_Step-{tpf_animation_step}_TPF_Animation_Exptime={exptime}s.mp4", writer=tpf_animation_writer)
    tpf_animation_end_time = time.time()  # measure the end time
    tpf_animation_rendering_time = tpf_animation_end_time - tpf_animation_start_time # calculate the rendering time
    print(f"Rendered {name} Sector {sector} Eleanor TPF animation in {tpf_animation_rendering_time:.3f} seconds.\n")
    config = update_config(config_path, {'eleanor.visualize.tpf_animation_rendering_time': round(tpf_animation_rendering_time, 3)})
else:
    print(f"Skipped {name} Sector {sector} Eleanor TPF animation rendering.\n")




### Light Curve ###
# Plot the all-in-one light curves
i += 1 # count the step
lcs_plot, ax_lcs = plt.subplots(figsize=(20, 5))
lcs_offset_list = config['eleanor']['visualize']['lcs_offset'] ##### set the offsets for different types of light curves #####
lc_raw.normalize().scatter(ax=ax_lcs, offset=lcs_offset_list[0], label=f"Raw, {cdpp_transit_duration:.3f}h-CDPP={lc_raw_cdpp:.2f} ppm, offset={lcs_offset_list[0]}", c='k')
lc_corr.normalize().scatter(ax=ax_lcs, offset=lcs_offset_list[1], label=f"Corrected, {cdpp_transit_duration:.3f}h-CDPP={lc_corr_cdpp:.2f} ppm, offset={lcs_offset_list[1]}", c='r')
lc_pca.normalize().scatter(ax=ax_lcs, offset=lcs_offset_list[2], label=f"PCA, {cdpp_transit_duration:.3f}h-CDPP={lc_pca_cdpp:.2f} ppm, offset={lcs_offset_list[2]}", c='g')
lc_psf.normalize().scatter(ax=ax_lcs, offset=lcs_offset_list[3], label=f"PSF, {cdpp_transit_duration:.3f}h-CDPP={lc_psf_cdpp:.2f} ppm, offset={lcs_offset_list[3]}", c='b')
ax_lcs.set_title(f"{name} Sector {sector} Eleanor All-in-one Light Curves Exptime={exptime}s")
lcs_plot.figure.tight_layout()
lcs_plot.figure.savefig(eleanor_processed_lightcurve_plots_dir_source_sector_pc + f"/{i:02}_{name}_Sector-{sector}_Eleanor_All-in-one_Light_Curves_Exptime={exptime}s.png")
plt.close()


# Define the lightcurve-specified directories based on the type of light curve
eleanor_processed_lightcurve_plots_dir_source_sector_pc_lc_list = []
for lc_type in lc_type_list:
    eleanor_processed_lightcurve_plots_dir_source_sector_pc_lc = eleanor_processed_lightcurve_plots_dir_source_sector_pc + f"/{lc_type}"
    os.makedirs(eleanor_processed_lightcurve_plots_dir_source_sector_pc_lc, exist_ok=True)
    eleanor_processed_lightcurve_plots_dir_source_sector_pc_lc_list.append(eleanor_processed_lightcurve_plots_dir_source_sector_pc_lc)


# Plot the individual light curve of the specific type
i += 1 # count the step
for l in range(len(lc_list)):
    lc = lc_list[l]
    lc_type = lc_type_list[l]
    eleanor_processed_lightcurve_plots_dir_source_sector_pc_lc = eleanor_processed_lightcurve_plots_dir_source_sector_pc_lc_list[l]

    lc_plot, ax_lc = plt.subplots(figsize=(20, 5))
    lc.normalize().scatter(ax=ax_lc, s=0.1)
    lc.normalize().errorbar(ax=ax_lc)
    ax_lc.set_title(f"{name} Sector {sector} Eleanor {lc_type} Light Curve Exptime={exptime}s")
    lc_plot.figure.tight_layout()
    lc_plot.figure.savefig(eleanor_processed_lightcurve_plots_dir_source_sector_pc_lc + f"/{i:02}-{l:01}_{name}_Sector-{sector}_Eleanor_{lc_type}_Light_Curve_Exptime={exptime}s.png")
    plt.close()


# Plot the pixel-by-pixel light curves
i += 1 # count the step
aperture_rows, aperture_cols = np.where(data.aperture > 0)
tpf_margin = config['eleanor']['visualize']['tpf_margin'] ##### set margin of the TPF plot #####
colrange = [int(np.min(aperture_cols) - tpf_margin), int(np.max(aperture_cols) + tpf_margin + 1)]
rowrange = [int(np.min(aperture_rows) - tpf_margin), int(np.max(aperture_rows) + tpf_margin + 1)]

j = 1 # count the sub-step
for l in range(len(lc_list)):
    lc_type = lc_type_list[l]
    eleanor_processed_lightcurve_plots_dir_source_sector_pc_lc = eleanor_processed_lightcurve_plots_dir_source_sector_pc_lc_list[l]

    if lc_type == 'Raw' or lc_type == 'Corrected':
        pbp_lc_plot = visualize.pixel_by_pixel(colrange=colrange, rowrange=rowrange, data_type=lc_type, color_by_pixel=True)
        pbp_lc_plot.suptitle(f"{name} Sector {sector} Eleanor Pixel-by-Pixel {lc_type}-flux Light Curves Exptime={exptime}s", fontsize='x-large')
        pbp_lc_plot.figure.tight_layout()
        pbp_lc_plot.figure.savefig(eleanor_processed_lightcurve_plots_dir_source_sector_pc_lc + f"/{i:02}-{j:01}-{l:01}_{name}_Sector-{sector}_Eleanor_Pixel-by-Pixel_{lc_type}-flux_Light_Curves_Exptime={exptime}s.png") # plot the pixel-by-pixel light curves of the selected type
        plt.close()

j += 1 # count the sub-step
pbp_pg_plot = visualize.pixel_by_pixel(colrange=colrange, rowrange=rowrange, data_type='amplitude', view='period', color_by_pixel=True)
pbp_pg_plot.suptitle(f"{name} Sector {sector} Eleanor Pixel-by-Pixel Amplitude (View='Period') Exptime={exptime}s", fontsize='x-large')
pbp_pg_plot.figure.tight_layout()
pbp_pg_plot.figure.savefig(eleanor_processed_lightcurve_plots_dir_source_sector_pc + f"/{i:02}-{j:01}_{name}_Sector-{sector}_Eleanor_Pixel-by-Pixel_Amplitude_(View='Period')_Exptime={exptime}s.png") # plot the pixel-by-pixel amplitudes (view='period')
plt.close()

j += 1 # count the sub-step
pbp_amp_plot = visualize.pixel_by_pixel(colrange=colrange, rowrange=rowrange, data_type='amplitude', view='frequency', color_by_pixel=True)
pbp_amp_plot.suptitle(f"{name} Sector {sector} Eleanor Pixel-by-Pixel Amplitude (View='Frequency') Exptime={exptime}s", fontsize='x-large')
pbp_amp_plot.figure.tight_layout()
pbp_amp_plot.figure.savefig(eleanor_processed_lightcurve_plots_dir_source_sector_pc + f"/{i:02}-{j:01}_{name}_Sector-{sector}_Eleanor_Pixel-by-Pixel_Amplitude_(View='Frequency')_Exptime={exptime}s.png") # plot the pixel-by-pixel amplitudes (view='frequency')
plt.close()




### Background ###
# Convert the 1D background to lightkurve.lightcurve objects
bkg_1d_pc = data.to_lightkurve(flux=data.flux_bkg)
bkg_1d_tpf = data.to_lightkurve(flux=data.tpf_flux_bkg)


# Plot the 1D background
i += 1 # count the step
bkg_1d_plot, ax_bkg_1d = plt.subplots(figsize=(20, 5))
bkg_1d_offset_list = config['eleanor']['visualize']['bkg_1d_offset'] ##### set the offsets for different types of 1D background #####
bkg_1d_pc.scatter(ax=ax_bkg_1d, normalize=False, offset=bkg_1d_offset_list[0], label=f"1D Postcard, offset={bkg_1d_offset_list[0]}", c='r')
bkg_1d_tpf.scatter(ax=ax_bkg_1d, normalize=False, offset=bkg_1d_offset_list[1], label=f"1D TPF, offset={bkg_1d_offset_list[1]}", c='g')
ax_bkg_1d.set_title(f"{name} Sector {sector} Eleanor 1D Background Exptime={exptime}s ({data.bkg_type} is applied.)")
bkg_1d_plot.figure.tight_layout()
bkg_1d_plot.figure.savefig(eleanor_processed_lightcurve_plots_dir_source_sector_pc + f"/{i:02}-{name}_Sector-{sector}_Eleanor_1D-Background_Exptime={exptime}s.png")
plt.close()


# Plot the 2D background, the aperture, the aperture-overplotted & Gaia-overlay TPF and the Gaia-overlay TPF
i += 1 # count the step
k = config['eleanor']['visualize']['cadence'] # set the cadence index
bkg_2d_aperture_tpf_plot, ((ax_bkg_2d_tpf, ax_aperture), (ax_aperture_overplotted_tpf, ax_gaia_overlay_tpf)) = plt.subplots(2, 2, figsize=(15, 10))
ax_bkg_2d_tpf.imshow(data.bkg_tpf[k], origin='lower')
if postcard:
    ax_bkg_2d_tpf.set_title(f"{name} Sector {sector} Eleanor 2D Interpolated Background\nCadence {k:04} Exptime={exptime}s (within the scope of TPF)") # plot the 2D background within the scope of TPF
elif tesscut:
    ax_bkg_2d_tpf.set_title(f"{name} Sector {sector} Eleanor TESScut Postcard\nCadence {k:04} Exptime={exptime}s") # the 2D background is the same as the TESScut postcard when using TESScut
ax_aperture.imshow(data.aperture, origin='lower')
ax_aperture.set_title(f"{name} Sector {sector} Eleanor Aperture Exptime={exptime}s") # plot the aperture
visualize.aperture_contour(ax=ax_aperture_overplotted_tpf, cadence=k, aperture=data.aperture)
visualize.add_gaia_figure_elements(fig_or_ax=ax_aperture_overplotted_tpf, individual=True, magnitude_limit=18) # add the Gaia figure elements
ax_aperture_overplotted_tpf.set_title(f"{name} Sector {sector} Eleanor Aperture-overplotted &\nGaia-overlay TPF Cadence {k:04} Exptime={exptime}s") # plot the aperture-overplotted and Gaia-overlay TPF
visualize.plot_gaia_overlay(ax=ax_gaia_overlay_tpf, cadence=k, magnitude_limit=18)
ax_gaia_overlay_tpf.set_title(f"{name} Sector {sector} Eleanor Gaia-overlay TPF\nCadence {k:04} Exptime={exptime}s") # plot the Gaia-overlay TPF
bkg_2d_aperture_tpf_plot.figure.tight_layout()
bkg_2d_aperture_tpf_plot.figure.savefig(eleanor_processed_lightcurve_plots_dir_source_sector_pc + f"/{i:02}-{name}_Sector-{sector}_Eleanor_2D-Background_&_Aperture_TPF_Exptime={exptime}s.png")
plt.close()


# Render the 2D background animation
i += 1 # count the step

# Set whether to render the 2D background animation
render_bkg_2d_animation = config['eleanor']['visualize']['render_bkg_2d_animation']
# validate the 'render_bkg_2d_animation' parameter
if tesscut and render_bkg_2d_animation:
    warnings.warn("Do not render the 2D background animation when using TESScut because the 2D background is the same as the TESScut postcard when using TESScut.\n"
                  "The 'render_bkg_2d_animation' parameter has been automatically set to 'false'.")
    render_bkg_2d_animation = False
    config = update_config(config_path, {'eleanor.visualize.render_bkg_2d_animation': False})

config = load_config(config_path)
render_bkg_2d_animation = config['eleanor']['visualize']['render_bkg_2d_animation'] ##### set whether to render the 2D background animation #####
bkg_2d_animation_step = config['eleanor']['visualize']['bkg_2d_animation_step'] ##### set the step size of the 2D background animation #####


if render_bkg_2d_animation:
    bkg_2d_animation_start_time = time.time()  # measure the start time
    print(f"Rendering the {name} Sector {sector} Eleanor 2D background animation...")
    bkg_2d_animation_framerate = 25 # set the framerate
    bkg_2d_animation_plot, ax_bkg_2d_animation = plt.subplots(figsize=(10, 10))
    bkg_2d_animation_artists = []
    for k in tqdm(range(0, len(data.bkg_tpf), bkg_2d_animation_step), desc="Rendering frames of 2D Background animation", unit=" frame"):
        ax_bkg_2d_animation_cadence_img = ax_bkg_2d_animation.imshow(data.bkg_tpf[k], origin='lower', animated=True)
        ax_bkg_2d_animation_cadence_title = ax_bkg_2d_animation.text(0.5, 1.01, f"{name} Sector {sector} Eleanor 2D Background (Cadence {k:04}) Exptime={exptime}s", ha='center', transform=ax_bkg_2d_animation.transAxes, fontfamily=rcParams['font.family'], fontsize='xx-large', fontweight=rcParams['axes.titleweight'])
        bkg_2d_animation_artists.append([ax_bkg_2d_animation_cadence_img, ax_bkg_2d_animation_cadence_title])
    print("Frames rendered, now compiling the animation...")
    bkg_2d_animation = animation.ArtistAnimation(fig=bkg_2d_animation_plot, artists=bkg_2d_animation_artists, interval=int(1000/bkg_2d_animation_framerate), blit=True)
    bkg_2d_animation_writer = animation.FFMpegWriter(fps=int(bkg_2d_animation_framerate), metadata=dict(artist='TELCEF'), bitrate=2000)
    bkg_2d_animation.save(filename=eleanor_processed_lightcurve_plots_dir_source_sector_pc + f"/{i:02}_{name}_Sector-{sector}_Eleanor_Step-{bkg_2d_animation_step}_2D_Background_Animation_Exptime={exptime}s.mp4", writer=bkg_2d_animation_writer)
    bkg_2d_animation_end_time = time.time()  # measure the end time
    bkg_2d_animation_rendering_time = bkg_2d_animation_end_time - bkg_2d_animation_start_time # calculate the rendering time
    print(f"Rendered {name} Sector {sector} Eleanor 2D Background animation in {bkg_2d_animation_rendering_time:.3f} seconds.\n")
    config = update_config(config_path, {'eleanor.visualize.bkg_2d_animation_rendering_time': round(bkg_2d_animation_rendering_time, 3)})
else:
    print(f"Skipped {name} Sector {sector} Eleanor 2D Background animation rendering.\n")




### Postcard ###
# Render the postcard animation
i += 1 # count the step

# Set whether to render the postcard animation
render_postcard_animation = config['eleanor']['visualize']['render_postcard_animation'] ##### set whether to render the postcard animation #####
postcard_animation_step = config['eleanor']['visualize']['postcard_animation_step']


if render_postcard_animation:
    postcard_animation_start_time = time.time()  # measure the start time
    print(f"Rendering the {name} Sector {sector} Eleanor Postcard animation...")
    postcard_animation_framerate = 25 # set the framerate
    if postcard:
        postcard_animation_plot, ax_postcard_animation = plt.subplots(figsize=(15, 10))
    elif tesscut:
        postcard_animation_plot, ax_postcard_animation = plt.subplots(figsize=(10, 10))
    postcard_animation_artists = []
    for k in tqdm(range(0, len(data.post_obj.flux), postcard_animation_step), desc="Rendering frames of Postcard animation", unit=" frame"):
        ax_postcard_animation_cadence_img = ax_postcard_animation.imshow(data.post_obj.flux[k], origin='lower', animated=True)
        ax_postcard_animation_cadence_title = ax_postcard_animation.text(0.5, 1.01, f"{name} Sector {sector} Eleanor Postcard (Cadence {k:04}) Exptime={exptime}s", ha='center', transform=ax_postcard_animation.transAxes, fontfamily=rcParams['font.family'], fontsize='xx-large', fontweight=rcParams['axes.titleweight'])
        postcard_animation_artists.append([ax_postcard_animation_cadence_img, ax_postcard_animation_cadence_title])
    print("Frames rendered, now compiling the animation...")
    postcard_animation = animation.ArtistAnimation(fig=postcard_animation_plot, artists=postcard_animation_artists, interval=int(1000/postcard_animation_framerate), blit=True)
    postcard_animation_writer = animation.FFMpegWriter(fps=int(postcard_animation_framerate), metadata=dict(artist='zzyu'), bitrate=2000)
    postcard_animation.save(filename=eleanor_processed_lightcurve_plots_dir_source_sector_pc + f"/{i:02}_{name}_Sector-{sector}_Eleanor_Step-{postcard_animation_step}_Postcard_Animation_Exptime={exptime}s.mp4", writer=postcard_animation_writer)
    postcard_animation_end_time = time.time()  # measure the end time
    postcard_animation_rendering_time = postcard_animation_end_time - postcard_animation_start_time  # calculate the rendering time
    print(f"Rendered {name} Sector {sector} Eleanor Postcard animation in {postcard_animation_rendering_time:.3f} seconds.\n")
    config = update_config(config_path, {'eleanor.visualize.postcard_animation_rendering_time': round(postcard_animation_rendering_time, 3)})
else:
    print(f"Skipped {name} Sector {sector} Eleanor Postcard animation rendering.\n")