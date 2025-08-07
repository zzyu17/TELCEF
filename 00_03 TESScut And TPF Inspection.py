import os
import lightkurve as lk
from astroquery.mast import Tesscut
import matplotlib.pyplot as plt




# Set the single sector and exposure time to be processed
sector = 54 ##### set the single sector to be processed #####
camera = 1 ##### set the camera number to be processed #####
ccd = 4 ##### set the ccd number to be processed #####
exptime = 600 ##### set the exposure time of data to be processed #####

# Define the starting point of TESS observation time
TESS_time = 2457000 # BJD


##### Define the source parameters #####
name = "WASP-80"
tic = 243921117
coords = (303.16737208333325,-2.144218611111111)
gaia = 4223507222112425344
tess_mag = 10.3622




# Define the directories
root = os.getcwd()
eleanor_root = os.path.expanduser("~/.eleanor")
eleanor_root_metadata = eleanor_root + f"/metadata"
eleanor_root_metadata_sector = eleanor_root_metadata + f"/s00{sector}"
eleanor_root_tesscut = eleanor_root + f"/tesscut"
eleanor_root_tesscut_target = eleanor_root_tesscut + f"/{name}"
lightkurve_root = os.path.expanduser("~/.lightkurve")
lightkurve_cache_root = os.path.expanduser("~/.lightkurve-cache")
lightkurve_cache_root_tesscut = lightkurve_cache_root + f"/tesscut"
tesscut_and_tpf_inspection_dir = root + "/00_03 TESScut And TPF Inspection"
os.makedirs(tesscut_and_tpf_inspection_dir, exist_ok=True)




# Define the method of obtaining the tesscut or tpf of the source and the type of tesscut or tpf
tpf_method = 'astroquery'  # 'lightkurve_search_tesscut' for lightkurve.search_tesscut(), 'lightkurve_search_targetpixelfile' for 'lightkurve.search_targetpixelfile()', 'read' for lightkurve.read() (FITS file already exists) or 'astroquery' for astroquery.mast.Tesscut.download_cutouts()
tpf_type = 'tesscut'  # 'tesscut' or 'tpf'

# Define the tesscut and tpf sizes
tesscut_size_1d = 31
tpf_height = 15
tpf_width = 15




# Search, download and/or read the tesscut or tpf of the source based on tpf_method and tpf_type
if tpf_method == 'lightkurve_search_tesscut' and tpf_type == 'tesscut':
    search_result = lk.search_tesscut(f'{name}', sector=sector)
    print(search_result)
    tpf = search_result.download(cutout_size=(tesscut_size_1d, tesscut_size_1d), path=lightkurve_cache_root_tesscut)
    tpf_plot_title = f"{name} Sector {sector} {tesscut_size_1d}x{tesscut_size_1d} TESScut_lightkurve"
    tpf_plot_fn = f"{name} Sector {sector} {tesscut_size_1d}x{tesscut_size_1d} TESScut_lightkurve.png"

elif tpf_method == 'lightkurve_search_targetpixelfile' and tpf_type == 'tpf':
    search_result = lk.search_targetpixelfile(f'{name}', sector=sector, exptime=exptime)
    print(search_result)
    tpf = search_result.download(cutout_size=(tpf_height, tpf_width), path=lightkurve_cache_root_tesscut)
    tpf_plot_title = f"{name} Sector {sector} {tpf_width}x{tpf_height} Exptime={exptime}s TPF_lightkurve"
    tpf_plot_fn = f"{name} Sector {sector} {tpf_width}x{tpf_height} Exptime={exptime}s TPF_lightkurve.png"


elif tpf_method == 'read':
    if tpf_type == 'tesscut':
        tpf = lk.read(eleanor_root_tesscut + f"/tess-s{sector:04}-{camera:01}-{ccd:01}_{coords[0]:.6f}_{coords[1]:.6f}_{tesscut_size_1d}x{tesscut_size_1d}_astrocut.fits")
        # tpf = lk.read(eleanor_root_tesscut + f"/tess-s0054-1-4_303.167372_-2.144219_31x31_astrocut.fits")
        tpf_plot_title = f"{name} Sector {sector} {tesscut_size_1d}x{tesscut_size_1d} TESScut"
        tpf_plot_fn = f"{name} Sector {sector} {tesscut_size_1d}x{tesscut_size_1d} TESScut.png"
    elif tpf_type == 'tpf':
        tpf = lk.read(eleanor_root_tesscut + f"/tess-s{sector:04}-{camera:01}-{ccd:01}_{coords[0]:.6f}_{coords[1]:.6f}_{tpf_height}x{tpf_width}_astrocut.fits")
        # tpf = lk.read(eleanor_root_tesscut + f"/tess-s0010-1-4_188.386850_-10.146174_15x15_astrocut.fits")
        tpf_plot_title = f"{name} Sector {sector} {tpf_width}x{tpf_height} TPF"
        tpf_plot_fn = f"{name} Sector {sector} {tpf_width}x{tpf_height} TPF.png"


elif tpf_method == 'astroquery':
    if tpf_type == 'tesscut':
        tpf_path_table = Tesscut.download_cutouts(objectname=f'{name}', size=(tesscut_size_1d, tesscut_size_1d), sector=sector, path=lightkurve_cache_root_tesscut)
        tpf = lk.read(tpf_path_table['Local Path'].data[0])
        tpf_plot_title = f"{name} Sector {sector} {tesscut_size_1d}x{tesscut_size_1d} TESScut_astroquery"
        tpf_plot_fn = f"{name} Sector {sector} {tesscut_size_1d}x{tesscut_size_1d} TESScut_astroquery.png"
    elif tpf_type == 'tpf':
        tpf_path_table = Tesscut.download_cutouts(objectname=f'{name}', size=(tpf_height, tpf_width), sector=sector, path=lightkurve_cache_root_tesscut)
        tpf = lk.read(tpf_path_table['Local Path'].data[0])
        tpf_plot_title = f"{name} Sector {sector} {tpf_width}x{tpf_height} TPF_astroquery"
        tpf_plot_fn = f"{name} Sector {sector} {tpf_width}x{tpf_height} TPF_astroquery.png"




# Plot the tesscut or tpf of the source
tpf_plot, ax_tpf = plt.subplots(figsize=(10, 10))
tpf.plot(ax=ax_tpf)
ax_tpf.set_title(tpf_plot_title)
tpf_plot.figure.tight_layout()
tpf_plot.figure.savefig(tesscut_and_tpf_inspection_dir + "/" + tpf_plot_fn)