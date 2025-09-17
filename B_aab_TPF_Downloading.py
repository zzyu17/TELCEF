import shutil
import os
import warnings

import lightkurve as lk
from astropy.io import fits
from astroquery.mast import Tesscut

from utils import remove_subfolder, search_matched_fits_file, format_fits_fn, remove_zip
from A_ab_Configuration_Loader import *




# Define the directories
tpf_downloaded_dir = data_dir + config['directory']['tpf_downloaded_dir']
os.makedirs(tpf_downloaded_dir, exist_ok=True)
tpf_downloaded_dir_source = tpf_downloaded_dir + f"/{name}"
os.makedirs(tpf_downloaded_dir_source, exist_ok=True)


# Define the downloading criteria
author = config['tpf_downloading_criteria']['author']
exptime = config['tpf_downloading_criteria']['exptime'] if config['tpf_downloading_criteria']['exptime'] is not None else exptime
tpf_height = config['tpf_downloading_criteria']['tpf_height']
tpf_width = config['tpf_downloading_criteria']['tpf_width']
# validate the exptime when author is "TESScut"
if author.lower() == "tesscut" and exptime not in [200, 600, 1800]:
    raise ValueError("When author is 'TESScut' (i.e., obtaining TPF via TESScut from TESS FFI), exptime must be one of [200, 600, 1800]. Please correct the exptime in the configuration file.")


# Merge the related configurations into one
if config['tpf_downloading_criteria']['exptime'] is None:
    matched_tpf_config = {**config['mission'], **config['tpf_downloading_criteria'], **config['source']}
else:
    matched_tpf_config = {**config['mission'], **config['source'], **config['tpf_downloading_criteria']}


# Define the filename and path
tpf_downloaded_fits_metadata_dict = {'type': 'tpf', 'name': name, 'mission': mission, 'sector': sector, 'author': author, 'exptime': exptime, 'tpf_height': tpf_height, 'tpf_width': tpf_width}
tpf_downloaded_fn = f"/{format_fits_fn(tpf_downloaded_fits_metadata_dict)}"
tpf_downloaded_path = tpf_downloaded_dir_source + tpf_downloaded_fn




# Check if the TPF file to be downloaded already exists in tpf_downloaded_dir_source
with warnings.catch_warnings(record=True) as w:
    matched_fits_file_path = search_matched_fits_file(tpf_downloaded_dir_source, 'tpf', matched_tpf_config)
if matched_fits_file_path is not None:
    # update the configurations if the matched TPF file doesn't have the expected size
    if w and "TPF size does not match" in str(w[-1].message):
        warnings.warn(w[-1].message)
        tpf_downloaded_hdulist = fits.open(matched_fits_file_path)
        tpf_downloaded_aperture_hdu = tpf_downloaded_hdulist[2]
        tpf_downloaded_aperture_header = tpf_downloaded_aperture_hdu.header
        tpf_width = tpf_downloaded_aperture_header.get('NAXIS1', None)
        tpf_height = tpf_downloaded_aperture_header.get('NAXIS2', None)
        update_dict = {'tpf_downloading_criteria.tpf_height': tpf_height, 'tpf_downloading_criteria.tpf_width': tpf_width}
        config = update_config(config_path, update_dict)

    print(f"TPF file already exists in the data directory: {matched_fits_file_path}.\n")


else:
    # Check if the TPF file to be downloaded already exists in the lightkurve_cache_root_tpf or lightkurve_cache_root_tesscut
    if "spoc" in author.lower():
        with warnings.catch_warnings(record=True) as w:
            matched_fits_file_path = search_matched_fits_file(lightkurve_cache_root_tpf, 'tpf', matched_tpf_config)
    elif author.lower() == "tesscut":
        matched_fits_file_path = search_matched_fits_file(lightkurve_cache_root_tesscut, 'tpf', matched_tpf_config)
    else:
        matched_fits_file_path = None

    if matched_fits_file_path is not None:
        # update the configurations and FITS filename if the matched TPF file doesn't have the expected size
        if w and "TPF size does not match" in str(w[-1].message):
            warnings.warn(w[-1].message)
            tpf_downloaded_hdulist = fits.open(matched_fits_file_path)
            tpf_downloaded_aperture_hdu = tpf_downloaded_hdulist[2]
            tpf_downloaded_aperture_header = tpf_downloaded_aperture_hdu.header
            tpf_width = tpf_downloaded_aperture_header.get('NAXIS1', None)
            tpf_height = tpf_downloaded_aperture_header.get('NAXIS2', None)
            update_dict = {'tpf_downloading_criteria.tpf_height': tpf_height, 'tpf_downloading_criteria.tpf_width': tpf_width}
            config = update_config(config_path, update_dict)

            tpf_downloaded_fits_metadata_dict = {'type': 'tpf', 'name': name, 'mission': mission, 'sector': sector, 'author': author, 'exptime': exptime, 'tpf_height': tpf_height, 'tpf_width': tpf_width}
            tpf_downloaded_fn = f"/{format_fits_fn(tpf_downloaded_fits_metadata_dict)}"
            tpf_downloaded_path = tpf_downloaded_dir_source + tpf_downloaded_fn

        print(f"TPF file found in Lightkurve cache: {matched_fits_file_path}.")

        # copy the matched FITS file to the tpf_downloaded_dir_source with the standardized filename
        shutil.copy(matched_fits_file_path, tpf_downloaded_path)
        print(f"Copied to the data directory of the source: {tpf_downloaded_path}.\n")


    # Download the TPF via Lightkurve if not found in both directories
    else:
        if "spoc" in author.lower():
            print(f"No existing TPF file found, downloading the {mission} Sector {sector} Author {author} Exptime={exptime}s TPF of {name} via Lightkurve...")
        elif author.lower() == "tesscut":
            print(f"No existing TPF file found, downloading the {mission} Sector {sector} Author {author} {tpf_width}x{tpf_height} Exptime={exptime}s TPF of {name} via TESScut...")

        # convert some specific metadata expressions
        if "spoc" in author.lower():
            author_alias = "*spoc*"
            tpf_downloaded_original = lk.search_targetpixelfile(name, mission=mission, sector=sector, author=author_alias, exptime=exptime).download(download_dir=tpf_downloaded_dir_source)
        elif author.lower() == "tesscut":
            tpf_downloaded_original_path_table = Tesscut.download_cutouts(objectname=name, size=(tpf_height, tpf_width), sector=sector, path=tpf_downloaded_dir_source, inflate=False)
            tpf_downloaded_original = lk.read(tpf_downloaded_original_path_table['Local Path'].data[0])

        if tpf_downloaded_original is not None:
            # check if the downloaded TPF has the expected size, if not, update the configurations and FITS filename
            tpf_downloaded_aperture_hdu = tpf_downloaded_original.hdu[2]
            tpf_downloaded_aperture_header = tpf_downloaded_aperture_hdu.header
            tpf_downloaded_width = tpf_downloaded_aperture_header.get('NAXIS1', None)
            tpf_downloaded_height = tpf_downloaded_aperture_header.get('NAXIS2', None)
            if tpf_downloaded_width != tpf_width or tpf_downloaded_height != tpf_height:
                warnings.warn(f"The TPF size does not match: expected {tpf_width}x{tpf_height}, but got {tpf_downloaded_width}x{tpf_downloaded_height}. The 'tpf_width' and 'tpf_height' parameters in the configuration file will be updated accordingly.")
                tpf_width = tpf_downloaded_width
                tpf_height = tpf_downloaded_height
                update_dict = {'tpf_downloading_criteria.tpf_height': tpf_height, 'tpf_downloading_criteria.tpf_width': tpf_width}
                config = update_config(config_path, update_dict)

                tpf_downloaded_fits_metadata_dict = {'type': 'tpf', 'name': name, 'mission': mission, 'sector': sector, 'author': author, 'exptime': exptime, 'tpf_height': tpf_height, 'tpf_width': tpf_width}
                tpf_downloaded_fn = f"/{format_fits_fn(tpf_downloaded_fits_metadata_dict)}"
                tpf_downloaded_path = tpf_downloaded_dir_source + tpf_downloaded_fn

            # rename the downloaded FITS file to the standardized filename
            shutil.copy(tpf_downloaded_original.path, tpf_downloaded_path)
            remove_subfolder(tpf_downloaded_dir_source)
            remove_zip(tpf_downloaded_dir_source)
            print(f"Successfully downloaded and saved to the data directory of the source: {tpf_downloaded_path}.\n")
        else:
            print(f"No TPF found for the specified criteria: source: {name}, mission: {mission}, sector: {sector}, author: {author}, exptime={exptime}s, size: {tpf_width}x{tpf_height}.\n")