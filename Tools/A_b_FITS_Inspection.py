import os

from utils import inspect_fits_file
from A_ab_Configuration_Loader import *




# Define the directories
fits_inspection_dir = base_dir + config['directory']['fits_inspection_dir']
os.makedirs(fits_inspection_dir, exist_ok=True)

tpf_downloaded_dir = data_dir + config['directory']['tpf_downloaded_dir']
os.makedirs(tpf_downloaded_dir, exist_ok=True)
tpf_downloaded_dir_source = tpf_downloaded_dir + f"/{name}"
os.makedirs(tpf_downloaded_dir_source, exist_ok=True)

lc_downloaded_dir = data_dir + config['directory']['lc_downloaded_dir']
os.makedirs(lc_downloaded_dir, exist_ok=True)
lc_downloaded_dir_source = lc_downloaded_dir + f"/{name}"
os.makedirs(lc_downloaded_dir_source, exist_ok=True)

eleanor_lc_dir = data_dir + config['directory']['eleanor_lc_dir']
os.makedirs(eleanor_lc_dir, exist_ok=True)
eleanor_lc_dir_source = eleanor_lc_dir + f"/{name}"
os.makedirs(eleanor_lc_dir_source, exist_ok=True)




lc_downloaded_dir = data_dir + config['directory']['lc_downloaded_dir']
os.makedirs(lc_downloaded_dir, exist_ok=True)
lc_downloaded_dir_source = lc_downloaded_dir + f"/{name}"
os.makedirs(lc_downloaded_dir_source, exist_ok=True)

# Inspect each FITS file in the assigned directory (including sub-directories)
for root, dirs, files in os.walk(lc_downloaded_dir_source): ##### set the directory to be inspected #####
    for file in files:
        if file.endswith(".fits"):
            fits_path = root + '/' + file
            print(f"Inspecting {fits_path}...")
            inspect_fits_file(fits_path, fits_inspection_dir)
print("")

# Insepct a specific FITS file
# fits_path = eleanor_root_tesscut + "/tess-s0054-1-4_303.167372_-2.144219_31x31_astrocut.fits" ##### set the specific FITS file to be inspected #####
# print(f"Inspecting {fits_path}...\n")
# inspect_fits_file(fits_path, fits_inspection_dir)




print("Successfully inspected the FITS file(s).\n")