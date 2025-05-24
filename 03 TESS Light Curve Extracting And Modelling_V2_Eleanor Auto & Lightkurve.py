import lightkurve as lk
import eleanor
import os, warnings, time
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import rcParams
from matplotlib import collections
import numpy as np

# Ignore all warnings
warnings.filterwarnings("ignore")




### ------ Preparations ------ ###
# Update the data of certain sectors
# eleanor.Update(sector=10)
# eleanor.Update(sector=54)

# Set the single sector and exposure time to be processed
sector = 10 ##### set the single sector to be processed #####
exptime = 1800   ##### set the exposure time of data to be processed #####


# Define the starting point of TESS observation time
TESS_time = 2457000 # BJD

# ##### Define the source parameters #####
# name = "WASP-80"
# tic = 243921117
# coords = (303.16737208333325,-2.144218611111111)
# gaia = 4223507222112425344
# tess_mag = 10.3622

##### Define the source parameters #####
name = "WASP-107"
tic = 429302040
coords = (188.38685041666665, -10.146173611111111)
gaia = 3578638842054261248
tess_mag = 10.418




# Define the method name based on the script name
script_path = os.path.abspath(__file__)
script_name = str(os.path.basename(script_path))
last_underscore_idx = script_name.rfind('_')
dot_idx = script_name.rfind('.')
and_idx = script_name.rfind('&')
if last_underscore_idx != -1 and dot_idx != -1 and and_idx != -1 and last_underscore_idx < and_idx < dot_idx:
    method = script_name[last_underscore_idx + 1: dot_idx]
    method_eleanor = script_name[last_underscore_idx + 1: and_idx - 1]
    method_lightkurve = script_name[and_idx + 2: dot_idx]
else:
    print("Unresolvable script name. Please define the method name manually.")
    method = "Eleanor Auto & Lightkurve" # define the method name manually if the script name is unresolvable
    method_eleanor = "Eleanor Auto"
    method_lightkurve = "Lightkurve"

# Define the directories
root = os.getcwd()
processed_lightcurve_plots_dir = root + f"/03 Processed Lightcurve Plots_V2_{method}"
os.makedirs(processed_lightcurve_plots_dir, exist_ok=True)
processed_lightcurve_plots_parent_dir = processed_lightcurve_plots_dir + f"/{name}_Sector {sector}"
os.makedirs(processed_lightcurve_plots_parent_dir, exist_ok=True)

eleanor_root = os.path.expanduser("~/.eleanor")
eleanor_root_metadata = eleanor_root + f"/metadata"
eleanor_root_metadata_sector = eleanor_root_metadata + f"/s00{sector}"
eleanor_root_tesscut = eleanor_root + f"/tesscut"
os.makedirs(eleanor_root_tesscut, exist_ok=True)

eleanor_root_targetdata = eleanor_root + f"/targetdata"
os.makedirs(eleanor_root_targetdata, exist_ok=True)
eleanor_root_targetdata_source = eleanor_root_targetdata + f"/{name}"
os.makedirs(eleanor_root_targetdata_source, exist_ok=True)




### ------ Eleanor ------ ###
### Preparations ###
# Retrieve the source
# run eleanor.Source with default parameters first to see whether postcard is used or not, then set the 'postcard' and 'tesscut_size_1d' parameters manually
postcard = False ##### set whether to use postcard or not  #####
postcard_size = (148, 104) if postcard else None
tesscut = not postcard
tesscut_size_1d = 31 ##### set the TESScut size #####
tesscut_size = (tesscut_size_1d, tesscut_size_1d) if tesscut else None

lc_type = 'Corrected' ##### select the type of light curve to be processed #####

# define the method-specified directory based on whether postcard or tesscut is used and the type of light curve to be processed
if postcard:
    processed_lightcurve_plots_method_parent_dir = processed_lightcurve_plots_parent_dir + f"/{method}_Postcard_{lc_type}"
    post_dir = eleanor_root
if tesscut:
    processed_lightcurve_plots_method_parent_dir = processed_lightcurve_plots_parent_dir + f"/{method}_TESScut_{lc_type}"
    post_dir = eleanor_root_tesscut
os.makedirs(processed_lightcurve_plots_method_parent_dir, exist_ok=True)

source = eleanor.Source(name=name, sector=sector, tc=tesscut, tesscut_size=tesscut_size_1d, post_dir=post_dir) ##### run eleanor.Source with default parameters first #####
print(f"Found TIC {source.tic} (Gaia {source.gaia}), with TESS magnitude {source.tess_mag}, RA {source.coords[0]}, and Dec {source.coords[1]}")

# Retrieve the target data
tpf_height = 15
tpf_width = 15
if postcard:
    bkg_size_1d = 31
if tesscut:
    bkg_size_1d = tesscut_size_1d
aperture_mode = 'normal'
cal_cadences = None
do_pca = True
do_psf = True
regressors = None

data = eleanor.TargetData(source, height=tpf_height, width=tpf_width, bkg_size=bkg_size_1d, aperture_mode=aperture_mode, cal_cadences=cal_cadences, do_pca=do_pca, do_psf=do_psf, regressors=regressors)
if postcard:
    data.save(output_fn=f"{name}_Sector {sector}_{method}_Postcard.fits", directory=eleanor_root_targetdata_source)
if tesscut:
    data.save(output_fn=f"{name}_Sector {sector}_{method}_TESScut.fits", directory=eleanor_root_targetdata_source)

# Create a visualize class object
visualize = eleanor.Visualize(data)
visualize_postcard = eleanor.Visualize(data.post_obj, obj_type="postcard")

##### Set whether to render the animations or not #####
render_tpf_animation = False
render_aperture_contour_tpf_animation = False # whether to render the TPF animation with aperture contour
if postcard:
    render_bkg_2d_animation = False
if tesscut:
    render_bkg_2d_animation = False # do not render the 2D background animation when using TESScut
render_postcard_animation = False


### TPF ###
# Render the TPF animation
i = 1 # count the step
if render_tpf_animation:
    tpf_animation_start_time = time.time()  # measure the start time
    if render_aperture_contour_tpf_animation:
        print(f"Rendering the {name} Sector {sector} {method_eleanor} TPF animation WITH aperture contour...")
    else:
        print(f"Rendering the {name} Sector {sector} {method_eleanor} TPF animation WITHOUT aperture contour...")
    tpf_animation_framerate = 25 # set the framerate
    tpf_animation_plot, ax_tpf_animation = plt.subplots(figsize=(10, 10))
    tpf_animation_artists = []
    for k in range(len(data.tpf)):
        if render_aperture_contour_tpf_animation:
            visualize.aperture_contour(ax=ax_tpf_animation, cadence=k, aperture=data.aperture)
            ax_tpf_animation_cadence_img = ax_tpf_animation.get_images()[-1] # get the newest plotted aperture-overplotted TPF
        else:
            ax_tpf_animation_cadence_img = ax_tpf_animation.imshow(data.tpf[k], origin='lower', animated=True)
        ax_tpf_animation_cadence_title = ax_tpf_animation.text(0.5, 1.01, f"{name} Sector {sector} {method_eleanor} Target Pixel File (Cadence {k:04})", ha='center', transform=ax_tpf_animation.transAxes, fontfamily=rcParams['font.family'], fontsize='xx-large', fontweight=rcParams['axes.titleweight'])
        tpf_animation_artists.append([ax_tpf_animation_cadence_img, ax_tpf_animation_cadence_title])
    tpf_animation = animation.ArtistAnimation(fig=tpf_animation_plot, artists=tpf_animation_artists, interval=int(1000/tpf_animation_framerate), blit=True)
    tpf_animation_writer = animation.FFMpegWriter(fps=int(tpf_animation_framerate), metadata=dict(artist='zzyu'), bitrate=2000)
    tpf_animation.save(filename=processed_lightcurve_plots_method_parent_dir + f"/{i:02} {name} Sector {sector} {method_eleanor} TPF Exptime={exptime}s.mp4", writer=tpf_animation_writer)
    tpf_animation_end_time = time.time()  # measure the end time
    tpf_animation_rendering_time = tpf_animation_end_time - tpf_animation_start_time # calculate the rendering time
    print(f"Rendered the {name} Sector {sector} {method_eleanor} TPF animation in {tpf_animation_rendering_time} seconds.")
if not render_tpf_animation:
    print(f"Skipped the {name} Sector {sector} {method_eleanor} TPF animation rendering.")


### Light Curve ###
# Convert the flux to lightkurve.lightcurve objects
i += 1 # count the step
quality_mask = 'Default' ##### set the quality mask #####
lc_raw = data.to_lightkurve(flux=data.raw_flux, quality_mask=data.quality)
lc_corr = data.to_lightkurve(flux=data.corr_flux, quality_mask=data.quality)
lc_pca = data.to_lightkurve(flux=data.pca_flux, quality_mask=data.quality)
lc_psf = data.to_lightkurve(flux=data.psf_flux, quality_mask=data.quality)

# Plot the light curve
j = 1 # count the sub-step
lc_plot, ax_lc = plt.subplots(figsize=(20, 5))
lc_raw.scatter(ax=ax_lc, normalize=True, offset=0.2, label='Raw', c='k')
lc_corr.scatter(ax=ax_lc, normalize=True, offset=0.1, label='Corrected', c='r')
lc_pca.scatter(ax=ax_lc, normalize=True, offset=0, label='PCA', c='g')
lc_psf.scatter(ax=ax_lc, normalize=True, offset=-0.1, label='PSF Modelled', c='b')
ax_lc.set_title(f"{name} Sector {sector} {method_eleanor} Light Curve")
lc_plot.figure.tight_layout()
lc_plot.figure.savefig(processed_lightcurve_plots_method_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {method_eleanor} Light Curves Exptime={exptime}s.png") # plot all the Eleanor light curves
j += 1 # count the sub-step
if lc_type.lower() == 'raw':
    lc_proc = lc_raw
if lc_type.lower() == 'corrected':
    lc_proc = lc_corr
if lc_type.lower() == 'pca':
    lc_proc = lc_pca
if lc_type.lower() == 'psf' or lc_type.lower() == 'psf modelled':
    lc_proc = lc_psf
lc_proc_plot, ax_lc_proc = plt.subplots(figsize=(20, 5))
lc_proc.normalize().scatter(ax=ax_lc_proc)
ax_lc_proc.set_title(f"{name} Sector {sector} {method_eleanor} {lc_type} Light Curve")
lc_proc_plot.figure.tight_layout()
lc_proc_plot.figure.savefig(processed_lightcurve_plots_method_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {method_eleanor} {lc_type} Light Curve Exptime={exptime}s.png") # plot the light curve of the selected type

# Plot the pixel-by-pixel light curves
i += 1 # count the step
aperture_rows, aperture_cols = np.where(data.aperture > 0)
# aperture_height = np.max(aperture_cols) - np.min(aperture_cols) + 1 # count the height of the aperture (x-direction)
# aperture_width = np.max(aperture_rows) - np.min(aperture_rows) + 1 # count the width of the aperture (y-direction)
tpf_margin = 3 ##### set the margin of the TPF plot #####
colrange = [int(np.min(aperture_cols) - tpf_margin), int(np.max(aperture_cols) + tpf_margin + 1)]
rowrange = [int(np.min(aperture_rows) - tpf_margin), int(np.max(aperture_rows) + tpf_margin + 1)]
j = 1 # count the sub-step
pbp_lc_proc_plot = visualize.pixel_by_pixel(colrange=colrange, rowrange=rowrange, data_type=lc_type, color_by_pixel=True)
pbp_lc_proc_plot.suptitle(f"{name} Sector {sector} {method_eleanor} {lc_type}-flux Pixel-by-Pixel Light Curves", fontsize='x-large')
pbp_lc_proc_plot.figure.tight_layout()
pbp_lc_proc_plot.figure.savefig(processed_lightcurve_plots_method_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {method_eleanor} {lc_type}-flux Pixel-by-Pixel Light Curves Exptime={exptime}s.png") # plot the pixel-by-pixel light curves of the selected type
j += 1 # count the sub-step
pbp_lc_pg_plot = visualize.pixel_by_pixel(colrange=colrange, rowrange=rowrange, data_type="periodogram", color_by_pixel=True)
pbp_lc_pg_plot.suptitle(f"{name} Sector {sector} {method_eleanor} Periodogram Pixel-by-Pixel Light Curves", fontsize='x-large')
pbp_lc_pg_plot.figure.tight_layout()
pbp_lc_pg_plot.figure.savefig(processed_lightcurve_plots_method_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {method_eleanor} Periodogram Pixel-by-Pixel Light Curves Exptime={exptime}s.png") # plot the periodogram pixel-by-pixel light curves
j += 1 # count the sub-step
pbp_lc_amp_plot = visualize.pixel_by_pixel(colrange=colrange, rowrange=rowrange, data_type="amplitude", view='frequency', color_by_pixel=True)
pbp_lc_amp_plot.suptitle(f"{name} Sector {sector} {method_eleanor} Amplitude Pixel-by-Pixel Light Curves", fontsize='x-large')
pbp_lc_amp_plot.figure.tight_layout()
pbp_lc_amp_plot.figure.savefig(processed_lightcurve_plots_method_parent_dir + f"/{i:02}-{j:01} {name} Sector {sector} {method_eleanor} Amplitude Pixel-by-Pixel Light Curves Exptime={exptime}s.png") # plot the amplitude pixel-by-pixel light curves


### Background ###
# Convert the 1D background to lightkurve.lightcurve objects
bkg_1d_pc = data.to_lightkurve(flux=data.flux_bkg)
bkg_1d_tpf = data.to_lightkurve(flux=data.tpf_flux_bkg)

# Plot the 1D background
i += 1 # count the step
bkg_1d_plot, ax_bkg_1d = plt.subplots(figsize=(20, 5))
bkg_1d_pc.scatter(ax=ax_bkg_1d, normalize=False, offset=2000, label='1D postcard', c='r')
bkg_1d_tpf.scatter(ax=ax_bkg_1d, normalize=False, offset=0, label='1D TPF', c='g')
ax_bkg_1d.set_title(f"{name} Sector {sector} {method_eleanor} 1D Background ({data.bkg_type} is applied.)")
bkg_1d_plot.figure.tight_layout()
bkg_1d_plot.figure.savefig(processed_lightcurve_plots_method_parent_dir + f"/{i:02} {name} Sector {sector} {method_eleanor} 1D Background Exptime={exptime}s.png")

# Plot the aperture-overplotted TPF, the Gaia-overlay TPF, the 2D background and the aperture
i += 1 # count the step
k = 0 # cadence index
bkg_2d_aperture_tpf_plot, ((ax_aperture_contour_tpf, ax_gaia_overlay_tpf), (ax_bkg_2d_tpf, ax_aperture)) = plt.subplots(2, 2, figsize=(15, 10))
visualize.aperture_contour(ax=ax_aperture_contour_tpf, cadence=k, aperture=data.aperture)
visualize.add_gaia_figure_elements(fig_or_ax=ax_aperture_contour_tpf, individual=True, magnitude_limit=18) # add the Gaia figure elements
ax_aperture_contour_tpf.set_title(f"{name} Sector {sector} {method_eleanor} Aperture-overplotted &\nGaia-overlay Target Pixel File (Cadence {k:04})") # plot the aperture-overplotted and Gaia-overlay TPF
visualize.plot_gaia_overlay(ax=ax_gaia_overlay_tpf, cadence=k, magnitude_limit=18)
ax_gaia_overlay_tpf.set_title(f"{name} Sector {sector} {method_eleanor} Gaia-overlay TPF (Cadence {k:04})") # plot the Gaia-overlay TPF
ax_bkg_2d_tpf.imshow(data.bkg_tpf[k], origin='lower')
if postcard:
    ax_bkg_2d_tpf.set_title(f"{name} Sector {sector} {method_eleanor} 2D Interpolated Background\n(Cadence {k:04}, within the scope of TPF)") # plot the 2D background within the scope of TPF
if tesscut:
    ax_bkg_2d_tpf.set_title(f"{name} Sector {sector} {method_eleanor} 2D Interpolated Background\n(Cadence {k:04}, the same as the TESScut Postcard)") # the 2D background is the same as the TESScut postcard when using TESScut
ax_aperture.imshow(data.aperture, origin='lower')
ax_aperture.set_title(f"{name} Sector {sector} {method_eleanor} Aperture") # plot the aperture
bkg_2d_aperture_tpf_plot.figure.tight_layout()
bkg_2d_aperture_tpf_plot.figure.savefig(processed_lightcurve_plots_method_parent_dir + f"/{i:02} {name} Sector {sector} {method_eleanor} 2D Background & Aperture TPF Exptime={exptime}s.png")

# Render the 2D background animation
i += 1 # count the step
if render_bkg_2d_animation and postcard:
    bkg_2d_animation_start_time = time.time()  # measure the start time
    print(f"Rendering the {name} Sector {sector} {method_eleanor} 2D background animation...")
    bkg_2d_animation_framerate = 25 # set the framerate
    bkg_2d_animation_plot, ax_bkg_2d_animation = plt.subplots(figsize=(10, 10))
    bkg_2d_animation_artists = []
    for k in range(len(data.bkg_tpf)):
        ax_bkg_2d_animation_cadence_img = ax_bkg_2d_animation.imshow(data.bkg_tpf[k], origin='lower', animated=True)
        ax_bkg_2d_animation_cadence_title = ax_bkg_2d_animation.text(0.5, 1.01, f"{name} Sector {sector} {method_eleanor} 2D Background (Cadence {k:04})", ha='center', transform=ax_bkg_2d_animation.transAxes, fontfamily=rcParams['font.family'], fontsize='xx-large', fontweight=rcParams['axes.titleweight'])
        bkg_2d_animation_artists.append([ax_bkg_2d_animation_cadence_img, ax_bkg_2d_animation_cadence_title])
    bkg_2d_animation = animation.ArtistAnimation(fig=bkg_2d_animation_plot, artists=bkg_2d_animation_artists, interval=int(1000/bkg_2d_animation_framerate), blit=True)
    bkg_2d_animation_writer = animation.FFMpegWriter(fps=int(bkg_2d_animation_framerate), metadata=dict(artist='zzyu'), bitrate=2000)
    bkg_2d_animation.save(filename=processed_lightcurve_plots_method_parent_dir + f"/{i:02} {name} Sector {sector} {method_eleanor} 2D Background Exptime={exptime}s.mp4", writer=bkg_2d_animation_writer)
    bkg_2d_animation_end_time = time.time()  # measure the end time
    bkg_2d_animation_rendering_time = bkg_2d_animation_end_time - bkg_2d_animation_start_time # calculate the rendering time
    print(f"Rendered the {name} Sector {sector} {method_eleanor} 2D background animation in {bkg_2d_animation_rendering_time} seconds.")
elif not render_bkg_2d_animation:
    print(f"Skipped the {name} Sector {sector} {method_eleanor} 2D background animation rendering.")
elif not postcard:
    print(f"The 2D background of postcard produced using TESScut is the same as the TESScut postcard itself. Please render the postcard animation instead to visualize the TESScut postcard.")


### Postcard ###
# Render the postcard animation
i += 1 # count the step
if render_postcard_animation:
    postcard_animation_start_time = time.time()  # measure the start time
    print(f"Rendering the {name} Sector {sector} {method_eleanor} Postcard animation...")
    postcard_animation_framerate = 25 # set the framerate
    if postcard:
        postcard_animation_plot, ax_postcard_animation = plt.subplots(figsize=(15, 10))
    if tesscut:
        postcard_animation_plot, ax_postcard_animation = plt.subplots(figsize=(10, 10))
    postcard_animation_artists = []
    for k in range(len(data.post_obj.flux)):
        ax_postcard_animation_cadence_img = ax_postcard_animation.imshow(data.post_obj.flux[k], origin='lower', animated=True)
        ax_postcard_animation_cadence_title = ax_postcard_animation.text(0.5, 1.01, f"{name} Sector {sector} {method_eleanor} Postcard (Cadence {k:04})", ha='center', transform=ax_postcard_animation.transAxes, fontfamily=rcParams['font.family'], fontsize='xx-large', fontweight=rcParams['axes.titleweight'])
        postcard_animation_artists.append([ax_postcard_animation_cadence_img, ax_postcard_animation_cadence_title])
    postcard_animation = animation.ArtistAnimation(fig=postcard_animation_plot, artists=postcard_animation_artists, interval=int(1000/postcard_animation_framerate), blit=True)
    postcard_animation_writer = animation.FFMpegWriter(fps=int(postcard_animation_framerate), metadata=dict(artist='zzyu'), bitrate=2000)
    postcard_animation.save(filename=processed_lightcurve_plots_method_parent_dir + f"/{i:02} {name} Sector {sector} {method_eleanor} Postcard Exptime={exptime}s.mp4", writer=postcard_animation_writer)
    postcard_animation_end_time = time.time()  # measure the end time
    postcard_animation_rendering_time = postcard_animation_end_time - postcard_animation_start_time  # calculate the rendering time
    print(f"Rendered the {name} Sector {sector} {method_eleanor} Postcard animation in {postcard_animation_rendering_time} seconds.")
if not render_postcard_animation:
    print(f"Skipped the {name} Sector {sector} {method_eleanor} Postcard animation rendering.")




### ------ Lightkurve ------ ###
# Flatten the lightcurve and plot the flatten light curve
i += 1 # count the step
flatten = True # set whether to flatten or not
flatten_window_proportion = 0.02
flatten_window_length = int(lc_corr.time.shape[0] * flatten_window_proportion)
if flatten_window_length % 2 == 0:
    flatten_window_length += 1 # the window length should be an odd number
lc_flatten = lc_corr.flatten(window_length=flatten_window_length)
lc_flatten_plot, ax_flatten = plt.subplots(figsize=(20, 5))
lc_flatten.errorbar(ax=ax_flatten)
ax_flatten.set_title(f"{name} Sector {sector} {method_lightkurve} Flatten Light Curve (Window Length: {flatten_window_length})")
lc_flatten_plot.figure.tight_layout()
lc_flatten_plot.figure.savefig(processed_lightcurve_plots_method_parent_dir + f"/{i:02} {name} Sector {sector} {flatten_window_proportion * 100}% Window {method_lightkurve} Flatten Light Curve Exptime={exptime}s.png")




### ------ Documentation ------ ###
# Print the methodologies and results
methodology_result_file = open(processed_lightcurve_plots_method_parent_dir + f"/{name} Sector {sector} {method} Methodologies And Results.txt", "w", encoding='utf-8')
methodology_result_file.write(f"{name} Sector {sector} {method} Methodologies And Results\n\n")
methodology_result_file.write(f"Eleanor {lc_type} --- Lightkurve flatten\n\n") ##### set the lightkurve methodologies manually #####
methodology_result_file.write("Target Information: \n"
                                    f"Name: {name}\n"
                                    f"Sector: {source.sector}\n"
                                    f"Camera: {source.camera}\n"
                                    f"CCD: {source.chip}\n"
                                    f"Exposure Time: {exptime}\n"
                                    f"TIC: {source.tic}\n"
                                    f"Coords: {source.coords}\n"
                                    f"Gaia: {source.gaia}\n"
                                    f"TESS Magnitude: {source.tess_mag}\n"
                                    f"Target Location On CCD: {(source.position_on_chip[0], source.position_on_chip[1])}\n"
                                    f"Target Location On Postcard/TESScut: {(data.cen_x, data.cen_y)}\n"
                                    f"Target Location On Target Pixel File: {(data.tpf_star_x, data.tpf_star_y)}\n\n")
methodology_result_file.write("Eleanor: \n"
                                    f"Postcard: {postcard}\n"
                                    f"Postcard Size: {postcard_size}\n"
                                    f"TESScut: {tesscut}\n"
                                    f"TESScut Size: {tesscut_size}\n"
                                    f"Target Pixel File Size: ({tpf_width}, {tpf_height})\n"
                                    f"Background Size: {bkg_size_1d}\n"
                                    f"2D Background Size: {data.bkg_tpf[0].shape[1], data.bkg_tpf[0].shape[0]}\n"
                                    f"Aperture Mode: {aperture_mode}\n"
                                    f"Calibrated Cadences: {cal_cadences}\n"
                                    f"Background Type: {data.bkg_type}\n"
                                    f"Do PCA: {do_pca}\n"
                                    f"Do PSF: {do_psf}\n"
                                    f"Regressors: {regressors}\n"
                                    f"Quality Mask: {quality_mask}\n\n")
if render_tpf_animation:
    methodology_result_file.write(f"Rendered the {name} Sector {sector} {method_eleanor} TPF animation in {tpf_animation_rendering_time} seconds.\n\n")
if render_bkg_2d_animation:
    methodology_result_file.write(f"Rendered the {name} Sector {sector} {method_eleanor} 2D background animation in {bkg_2d_animation_rendering_time} seconds.\n\n")
if render_postcard_animation:
    methodology_result_file.write(f"Rendered the {name} Sector {sector} {method_eleanor} Postcard animation in {postcard_animation_rendering_time} seconds.\n\n")
methodology_result_file.write("Lightkurve: \n"
                                    f"Flatten: {flatten}\n"
                                    f"Flatten Window Proportion: {flatten_window_proportion}\n")
methodology_result_file.close()