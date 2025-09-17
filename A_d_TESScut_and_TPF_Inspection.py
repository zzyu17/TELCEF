import os

import lightkurve as lk
from astroquery.mast import Tesscut
import matplotlib.pyplot as plt

from A_ab_Configuration_Loader import *




# Define the directories
tesscut_and_tpf_inspection_dir = base_dir + config['directory']['tesscut_and_tpf_inspection_dir']
os.makedirs(tesscut_and_tpf_inspection_dir, exist_ok=True)




##### Define the method of obtaining the tesscut or tpf of the source and the type of tesscut or tpf #####
tpf_method = config['tesscut_and_tpf_inspection']['tpf_method']
tpf_type = config['tesscut_and_tpf_inspection']['tpf_type']

##### Define the tesscut and tpf sizes #####
tesscut_size_1d = config['tesscut_and_tpf_inspection']['tesscut_size_1d']
tpf_height = config['tesscut_and_tpf_inspection']['tpf_height']
tpf_width = config['tesscut_and_tpf_inspection']['tpf_width']




# Search, download and/or read the tesscut or tpf of the source based on tpf_method and tpf_type
if tpf_method == 'search_tesscut' and tpf_type == 'tesscut':
    search_result = lk.search_tesscut(name, sector=sector)
    tpf = search_result.download(cutout_size=(tesscut_size_1d, tesscut_size_1d), download_dir=lightkurve_cache_root)
    tpf_plot_title = f"{name} Sector {sector} {tesscut_size_1d}x{tesscut_size_1d} TESScut by Lightkurve"
    tpf_plot_fn = f"{name}_Sector-{sector}_{tesscut_size_1d}x{tesscut_size_1d}_TESScut_Lightkurve.png"

elif tpf_method == 'search_targetpixelfile' and tpf_type == 'tpf':
    search_result = lk.search_targetpixelfile(name, sector=sector, exptime=exptime)
    tpf = search_result.download(cutout_size=(tpf_height, tpf_width), download_dir=lightkurve_cache_root_tpf)
    tpf_plot_title = f"{name} Sector {sector} {tpf_width}x{tpf_height} Exptime={exptime}s TPF by Lightkurve"
    tpf_plot_fn = f"{name}_Sector-{sector}_{tpf_width}x{tpf_height}_Exptime={exptime}s_TPF_Lightkurve.png"

elif tpf_method == 'search_tesscut' and tpf_type == 'tpf':
    raise ValueError("tpf_type cannot be 'tpf' when tpf_method is 'search_tesscut'. Please use 'search_targetpixelfile' instead.")

elif tpf_method == 'search_targetpixelfile' and tpf_type == 'tesscut':
    raise ValueError("tpf_type cannot be 'tesscut' when tpf_method is 'search_targetpixelfile'. Please use 'search_tesscut' instead.")


elif tpf_method == 'read':
    if tpf_type == 'tesscut':
        tpf = lk.read(eleanor_root_tesscut + f"/tess-s{sector:04}-{camera:01}-{ccd:01}_{coords[0]:.6f}_{coords[1]:.6f}_{tesscut_size_1d}x{tesscut_size_1d}_astrocut.fits")
        # tpf = lk.read(eleanor_root_tesscut + f"/tess-s0054-1-4_303.167372_-2.144219_31x31_astrocut.fits")
        tpf_plot_title = f"{name} Sector {sector} {tesscut_size_1d}x{tesscut_size_1d} TESScut"
        tpf_plot_fn = f"{name}_Sector-{sector}_{tesscut_size_1d}x{tesscut_size_1d}_TESScut.png"
    elif tpf_type == 'tpf':
        tpf = lk.read(eleanor_root_tesscut + f"/tess-s{sector:04}-{camera:01}-{ccd:01}_{coords[0]:.6f}_{coords[1]:.6f}_{tpf_width}x{tpf_height}_astrocut.fits")
        # tpf = lk.read(eleanor_root_tesscut + f"/tess-s0010-1-4_188.386850_-10.146174_15x15_astrocut.fits")
        tpf_plot_title = f"{name} Sector {sector} {tpf_width}x{tpf_height} TPF"
        tpf_plot_fn = f"{name}_Sector-{sector}_{tpf_width}x{tpf_height}_TPF.png"


elif tpf_method == 'astroquery_tesscut':
    if tpf_type == 'tesscut':
        tpf_path_table = Tesscut.download_cutouts(objectname=name, size=(tesscut_size_1d, tesscut_size_1d), sector=sector, path=lightkurve_cache_root_tesscut)
        tpf = lk.read(tpf_path_table['Local Path'].data[0])
        tpf_plot_title = f"{name} Sector {sector} {tesscut_size_1d}x{tesscut_size_1d} TESScut by astroquery"
        tpf_plot_fn = f"{name}_Sector-{sector}_{tesscut_size_1d}x{tesscut_size_1d}_TESScut_astroquery.png"
    elif tpf_type == 'tpf':
        tpf_path_table = Tesscut.download_cutouts(objectname=name, size=(tpf_height, tpf_width), sector=sector, path=lightkurve_cache_root_tesscut)
        tpf = lk.read(tpf_path_table['Local Path'].data[0])
        tpf_plot_title = f"{name} Sector {sector} {tpf_width}x{tpf_height} TPF by astroquery"
        tpf_plot_fn = f"{name}_Sector-{sector}_{tpf_width}x{tpf_height}_TPF_astroquery.png"




# Plot the tesscut or tpf of the source
tpf_plot, ax_tpf = plt.subplots(figsize=(10, 10))
tpf.plot(ax=ax_tpf)
ax_tpf.set_title(tpf_plot_title)
tpf_plot.figure.tight_layout()
tpf_plot.figure.savefig(tesscut_and_tpf_inspection_dir + "/" + tpf_plot_fn)
plt.close()




print(f"Successfully inspected the {tpf_type} of {name} in Sector {sector} via the method of {tpf_method}.\n")