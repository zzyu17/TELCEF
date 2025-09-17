import shutil
import os

import lightkurve as lk

from utils import search_matched_fits_file, format_fits_fn
from A_ab_Configuration_Loader import *




# Define the directories
lc_downloaded_dir = data_dir + config['directory']['lc_downloaded_dir']
os.makedirs(lc_downloaded_dir, exist_ok=True)
lc_downloaded_dir_source = lc_downloaded_dir + f"/{name}"
os.makedirs(lc_downloaded_dir_source, exist_ok=True)


# Define the downloading criteria
author = config['lightcurve_downloading_criteria']['author']
exptime = config['lightcurve_downloading_criteria']['exptime'] if config['lightcurve_downloading_criteria']['exptime'] is not None else exptime


# Merge the related configurations into one
if config['lightcurve_downloading_criteria']['exptime'] is None:
    matched_lc_config = {**config['mission'], **config['lightcurve_downloading_criteria'], **config['source']}
else:
    matched_lc_config = {**config['mission'], **config['source'], **config['lightcurve_downloading_criteria']}


# Define the filename and path
lc_downloaded_fits_metadata_dict = {'type': 'lc', 'name': name, 'mission': mission, 'sector': sector, 'author': author, 'exptime': exptime}
lc_downloaded_fn = f"/{format_fits_fn(lc_downloaded_fits_metadata_dict)}"
lc_downloaded_path = lc_downloaded_dir_source + lc_downloaded_fn




# Check if the lightcurve file to be downloaded already exists in lc_downloaded_dir_source
matched_fits_file_path = search_matched_fits_file(lc_downloaded_dir_source, 'lc', matched_lc_config)
if matched_fits_file_path is not None:
    print(f"Lightcurve file already exists in the data directory: {matched_fits_file_path}.\n")


else:
    # Check if the lightcurve file to be downloaded already exists in the lightkurve_cache_root_lc
    matched_fits_file_path = search_matched_fits_file(lightkurve_cache_root_lc, 'lc', matched_lc_config)

    if matched_fits_file_path is not None:
        print(f"Lightcurve file found in Lightkurve cache: {matched_fits_file_path}.")
        # copy the matched FITS file to the lc_downloaded_dir_source with the standardized filename
        shutil.copy(matched_fits_file_path, lc_downloaded_path)
        print(f"Copied to the data directory of the source: {lc_downloaded_path}.\n")


    # Download the lightcurve via Lightkurve if not found in both directories
    else:
        print(f"No existing lightcurve file found, downloading the {mission} Sector {sector} Author {author} Exptime={exptime}s lightcurve of {name} via Lightkurve...")

        # convert some specific metadata expressions
        if "spoc" in author.lower():
            author_alias = "*spoc*"
            lc_downloaded_original = lk.search_lightcurve(name, mission=mission, sector=sector, author=author_alias, exptime=exptime).download(download_dir=lc_downloaded_dir_source)
        else:
            lc_downloaded_original = lk.search_lightcurve(name, mission=mission, sector=sector, author=author, exptime=exptime).download(download_dir=lc_downloaded_dir_source)

        if lc_downloaded_original is not None:
            # rename the downloaded FITS file to the standardized filename
            os.rename(lc_downloaded_original.path, lc_downloaded_path)
            print(f"Successfully downloaded and saved to the data directory of the source: {lc_downloaded_path}.\n")
        else:
            print(f"No lightcurve found for the specified criteria: source: {name}, mission: {mission}, sector: {sector}, author: {author}, exptime={exptime}s.\n")