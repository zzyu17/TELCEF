import os
import lightkurve as lk
from astroquery.mast import Tesscut
import matplotlib.pyplot as plt




# Set the single sector and exposure time to be processed
sector = 54 ##### set the single sector to be processed #####
exptime = 600 ##### set the exposure time of data to be processed #####

# Define the starting point of TESS observation time
TESS_time = 2457000 # BJD

##### Define the target parameters #####
name = "WASP-80"
tic = 243921117
coords = (303.16737208333325,-2.144218611111111)
gaia = 4223507222112425344
tess_mag = 10.3622

# ##### Define the target parameters #####
# name = "WASP-107"
# tic = 429302040
# coords = (188.38685041666665, -10.146173611111111)
# gaia = 3578638842054261248
# tess_mag = 10.418

# Define the tesscut and tpf sizes
tesscut_size_1d = 31
tpf_height = 15
tpf_width = 15

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




# Search, download and/or read the tesscut or tpf of the target

# # Utilize lightkurve.search_tesscut() or lightkurve.search_targetpixelfile()
# search_result = lk.search_targetpixelfile(f'TIC {tic}', sector=sector)
# print(search_result)
# tpf = search_result.download()
# print(tpf.shape)

# # Utilize lightkurve.read() (.fits file already exists)
# tpf = lk.read(eleanor_root_tesscut+"/tess-s0081-1-4_303.167372_-2.144219_31x31_astrocut.fits")

# Utilize astroquery.mast.Tesscut.download_cutouts()
tpf_path_table = Tesscut.download_cutouts(objectname=f'TIC {tic}', size=(tesscut_size_1d, tesscut_size_1d), sector=sector, path=lightkurve_cache_root_tesscut)
tpf = lk.read(tpf_path_table['Local Path'].data[0])




# Plot the tesscut or tpf of the target
tpf_plot, ax_tpf = plt.subplots(figsize=(10, 10))
tpf.plot(ax=ax_tpf)

# ax_tpf.set_title(f"{name} Sector {sector} {tesscut_size_1d}x{tesscut_size_1d} TESScut_lightkurve")
# ax_tpf.set_title(f"{name} Sector {sector} {tpf_width}x{tpf_height} TPF_lightkurve")
# ax_tpf.set_title(f"{name} Sector {sector} SPOC TPF_lightkurve")
# ax_tpf.set_title(f"{name} Sector {sector} {tesscut_size_1d}x{tesscut_size_1d} TESScut")
# ax_tpf.set_title(f"{name} Sector {sector} {tpf_width}x{tpf_height} TPF")
ax_tpf.set_title(f"{name} Sector {sector} {tesscut_size_1d}x{tesscut_size_1d} TESScut_astroquery")
# ax_tpf.set_title(f"{name} Sector {sector} {tpf_width}x{tpf_height} TPF_astroquery")

tpf_plot.figure.tight_layout()

# tpf_plot.figure.savefig(tesscut_and_tpf_inspection_dir + f"/{name} Sector {sector} {tesscut_size_1d}x{tesscut_size_1d} TESScut_lightkurve.png")
# tpf_plot.figure.savefig(tesscut_and_tpf_inspection_dir + f"/{name} Sector {sector} {tpf_width}x{tpf_height} TPF_lightkurve.png")
# tpf_plot.figure.savefig(tesscut_and_tpf_inspection_dir + f"/{name} Sector {sector} SPOC TPF_lightkurve.png")
# tpf_plot.figure.savefig(tesscut_and_tpf_inspection_dir + f"/{name} Sector {sector} {tesscut_size_1d}x{tesscut_size_1d} TESScut.png")
# tpf_plot.figure.savefig(tesscut_and_tpf_inspection_dir + f"/{name} Sector {sector} {tpf_width}x{tpf_height} TPF.png")
tpf_plot.figure.savefig(tesscut_and_tpf_inspection_dir + f"/{name} Sector {sector} {tesscut_size_1d}x{tesscut_size_1d} TESScut_astroquery.png")
# tpf_plot.figure.savefig(tesscut_and_tpf_inspection_dir + f"/{name} Sector {sector} {tpf_width}x{tpf_height} TPF_astroquery.png")