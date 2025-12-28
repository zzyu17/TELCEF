import shutil
import sys
import os

import lightkurve as lk
import pandas as pd
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

telcef_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),  "..", ".."))
sys.path = [p for p in sys.path if p != telcef_root]
sys.path.insert(0, telcef_root)
from utils import format_fits_fn, update_fits_headers, sort_lc, estimate_lc_flux_err
from A_ab_Configuration_Loader import *




# Define the author
author = "LWTIM-H400"


# Define the directories
lc_custom_dir = data_dir + "/Custom_Light_Curve"
os.makedirs(lc_custom_dir, exist_ok=True)
lc_custom_dir_source = lc_custom_dir + f"/{name}"
os.makedirs(lc_custom_dir_source, exist_ok=True)
lc_custom_dir_source_lwtim_H400 = lc_custom_dir_source + f"/LWTIM-H400"
os.makedirs(lc_custom_dir_source, exist_ok=True)

lc_downloaded_dir = data_dir + config['directory']['lc_downloaded_dir']
os.makedirs(lc_downloaded_dir, exist_ok=True)
lc_downloaded_dir_source = lc_downloaded_dir + f"/{name}"
os.makedirs(lc_downloaded_dir_source, exist_ok=True)




# Read the custom lightcurve CSV file via Lightkurve and save it into a FITS file to the downloaded lightcurve directory
lc_custom_list = []
lc_custom_csv_raw_path_list = [os.path.join(root, f) for root, _, files in os.walk(lc_custom_dir_source_lwtim_H400) for f in files if f.endswith('_raw.csv')]
transit_mask_indices_list = [(12, 21), (44, 53), (16, 25)] ##### set the transit mask start and end indices for each lightcurve #####
for l in range(len(lc_custom_csv_raw_path_list)):
    lc_custom_csv_raw_path = lc_custom_csv_raw_path_list[l]
    lc_custom_df = pd.read_csv(lc_custom_csv_raw_path).dropna()
    lc_custom_df.set_index(lc_custom_df.columns[0], inplace=True, drop=True)
    lc_custom_df.index.name = "index" # set the first column as the index and rename it
    print(f"Successfully found and read the custom lightcurve file: {lc_custom_csv_raw_path}.\n")


    # Retrieve the lightcurve
    lc_custom_df.sort_values('JD', ascending=True, inplace=True)
    lc_custom_btjd = (lc_custom_df['JD'].values - tess_time) * u.day
    lc_custom_gmag = lc_custom_df['Gmag'].values
    lc_custom_gmag0 = np.nanmedian(lc_custom_gmag) # calculate the reference Gaia magnitude
    lc_custom_flux_normalized = 10 ** (- (lc_custom_gmag - lc_custom_gmag0) / 2.512) # convert Gaia magnitude to normalized flux
    lc_custom = lk.TessLightCurve(time=lc_custom_btjd, flux=lc_custom_flux_normalized, **{'targetid':tic, 'centroid_col':np.nan, 'centroid_row':np.nan}) # Convert to TESSLightCurve object with dummy targetid, centroid_col, and centroid_row
    lc_custom_list.append(lc_custom)

    transit_mask = np.zeros(len(lc_custom.time), dtype=bool)
    transit_mask[transit_mask_indices_list[l][0]:transit_mask_indices_list[l][1]] = True
    lc_custom = estimate_lc_flux_err(lc_custom, mask=transit_mask, method='std_diff', new_column=True)
    lc_custom = estimate_lc_flux_err(lc_custom, mask=transit_mask, method='std', new_column=True)


    # Store the data in the dataframe and save it into a CSV file
    lc_custom_df['transit_mask'] = transit_mask
    lc_custom_df['flux'] = lc_custom.flux
    lc_custom_df['std_diff_flux'] = lc_custom['std_diff_flux']
    lc_custom_df['std_diff_flux_err'] = lc_custom['std_diff_flux_err']
    lc_custom_df['std_flux'] = lc_custom['std_flux']
    lc_custom_df['std_flux_err'] = lc_custom['std_flux_err']

    len_csv_float = max([len(str(d).split('.')[-1]) for d in lc_custom_df.select_dtypes(include=['float']).values.flatten()])
    lc_custom_csv_path = lc_custom_csv_raw_path.replace("_raw.csv", ".csv")
    lc_custom_df.to_csv(lc_custom_csv_path, index=True, float_format=f'%.{len_csv_float}f')
    print(f"Successfully retrieved the lightcurve and saved the data into a CSV file: {lc_custom_csv_path}.\n")

    fig1, ax1 = plt.subplots()
    lc_custom = lc_custom.select_flux(flux_column='flux', flux_err_column='std_diff_flux_err')
    lc_custom.scatter(ax=ax1)
    lc_custom.errorbar(ax=ax1, label=f"{sectors[l]}_std_diff")
    fig1.figure.savefig(lc_custom_csv_path.replace(".csv", "_std_diff.png"))

    fig2, ax2 = plt.subplots()
    lc_custom = lc_custom.select_flux(flux_column='flux', flux_err_column='std_flux_err')
    lc_custom.scatter(ax=ax2)
    lc_custom.errorbar(ax=ax2, label=f"{sectors[l]}_std")
    fig2.figure.savefig(lc_custom_csv_path.replace(".csv", "_std.png"))
    plt.close()


    # Save the lightcurve into a FITS file
    lc_downloaded_fits_metadata_dict = {'type':'lc', 'name':name, 'mission':mission, 'sector':sectors[l], 'author':author, 'exptime':exptime}
    lc_downloaded_fn = f"/{format_fits_fn(lc_downloaded_fits_metadata_dict)}"
    lc_downloaded_path = lc_downloaded_dir_source + lc_downloaded_fn
    lc_custom.to_fits(path=lc_downloaded_path, overwrite=True)
    lc_downloaded_headers_update_dict = {0: {'TICID': (tic, "unique tess target identifier"), 'KEPLERID': None,
                                            'RA_OBJ': RA, 'DEC_OBJ': Dec,
                                            'MISSION': 'TESS', 'TELESCOP': 'TESS', 'INSTRUME': 'H400', # Cheat Lightkurve by setting 'MISSION' and 'TELESCOP' to "TESS" while marking the real instrument by setting 'INSTRUME' to "H400"
                                            'SECTOR': (sectors[l], "Observing sector"), 'CAMERA': (camera, "Camera number"), 'CCD': (ccd, "CCD chip number"),
                                            'ORIGIN': 'LWTIM-H400', 'CREATOR': 'TELCEF-LightCurve',
                                            'EXPTIME': (exptime, "exposure time in seconds")}}
    update_fits_headers(lc_downloaded_path, lc_downloaded_headers_update_dict)
    print(f"Successfully saved the lightcurve into a FITS file to the data directory of the source: {lc_downloaded_path}.\n\n")