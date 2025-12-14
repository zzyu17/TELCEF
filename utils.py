import yaml
from ruamel.yaml import YAML
import shutil
import re
import numbers
from collections.abc import Mapping, Iterable
import os
import sys
import subprocess
import warnings
import time

import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astroquery.mast import Observations
from tess_stars2px import tess_stars2px_function_entry as tess_stars2px
import lightkurve as lk
from eleanor.mast import coords_from_name, tic_from_coords, gaia_from_coords
from pytransit import QuadraticModel, RoadRunnerModel
import emcee
import numpy as np
import corner
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec




### ------ Mission-specified ------ ###
# Define the attribute map and the stage map for different missions
attribute_map = {
    'Kepler': 'quarter',
    'K2': 'campaign',
    'TESS': 'sector'
}

stage_map = {
    'Kepler': 'Quarter',
    'K2': 'Campaign',
    'TESS': 'Sector'
}




### ------ TELCEF Runner ------ ###
def run_script(script_name, args=None, max_retries=0, retry_delay=5.0):
    """
    Run a Python script with optional arguments in a command-line subprocess, with retry mechanism on failure.

    Parameters
    ----------
    script_name : str
        The name of the Python script to run.
    args : list of str, optional
        A list of arguments to pass to the command-line subprocess. Default is `None`.
    max_retries : int, optional
        The maximum number of retries on failure. Default is `0` (i.e., no retries).
    retry_delay : float, optional
        The delay (in seconds) before each retry. Default is `5.0`.

    Returns
    -------
    `True` if the script ran successfully, `False` otherwise.
    """
    print(f">>> Running: {script_name}...\n")

    # Construct the command to run the script
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)

    retry = 0
    while True:
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            for line in process.stdout:
                print(line.rstrip("\n"), flush=True)
            process.wait()
        except KeyboardInterrupt:
            if process and process.poll() is None:
                process.terminate()
            raise

        # If succeeded, return True
        if process.returncode == 0:
            print(f"\u2713 Completed running {script_name}.\n\n")
            return True

        # If failed, check whether to retry
        print(f"\u2715 Failed running {script_name}.\n")
        if retry >= max_retries:
            print(f"Exceeded max retries ({max_retries}). Given up on {script_name}.\n\n")
            return False

        # Sleep before retrying
        retry += 1
        print(f"Retrying ({retry}/{max_retries}) in {retry_delay} seconds...\n")
        time.sleep(retry_delay)




### ------ Directory and File Management ------ ###
def remove_subfolder(dir):
    """
    Remove all sub-folders and the files in them in a directory, but keep other files outside the subfolders.
    """
    for item in os.listdir(dir):
        item_path = os.path.join(dir, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)


def remove_zip(dir):
    """
    Remove all zip files in a directory.
    """
    for item in os.listdir(dir):
        item_path = os.path.join(dir, item)
        if os.path.isfile(item_path) and os.path.splitext(item)[-1].lower() == '.zip':
            os.remove(item_path)




### ------ Configuration And Dictionary Handling ------ ###
def load_config(config_path):
    """
    Load configurations from a YAML configuration file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def update_config(config_path, dict_update, decimal=True, precision=16):
    """
    Update the specific key-value pairs in a YAML configuration file with new ones,
    while preserving the original formatting, comments and other contents.
    Supports updating specific indices in lists and nested lists.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    dict_update : dict
        Dictionary where keys are dot-separated key paths and values are the new values.
        For updating specific list indices, use the format: 'key.path[index]' or 'key.path[index].nested_key'.
    decimal : bool, optional
        Whether to use decimal notation for floats. If `False`, scientific notation is used. Default is `True`.
    precision : int, optional
        Precision for float representation. Default is `16`.
    """
    yaml = YAML()

    # Set YAML formatting options
    yaml.preserve_quotes = True  # preserve the quote style
    yaml.indent(mapping=2, sequence=4, offset=2)  # preserve the indentation format

    # Customize float representation
    # set custom float representers
    def decimal_float_representer(dumper, value):
        if value == 0:
            # represent zero as '0.0'
            formatted = '0.0'
        else:
            # decimal notation with specified precision, removing trailing zeros
            formatted = f"{value:.{precision}f}"
            if '.' in formatted:
                while formatted.endswith('0'):
                    formatted = formatted[:-1]
                if formatted.endswith('.'):
                    formatted = formatted[:-1]
        return dumper.represent_scalar('tag:yaml.org,2002:float', formatted)
    def scientific_float_representer(dumper, value):
        # scientific notation with specified precision
        formatted = f"{value:.{precision}e}"
        return dumper.represent_scalar('tag:yaml.org,2002:float', formatted)
    # add custom float representers
    if decimal:
        yaml.representer.add_representer(float, decimal_float_representer)
    else:
        yaml.representer.add_representer(float, scientific_float_representer)

    # Load the existing configuration
    with open(config_path, 'r') as f:
        config = yaml.load(f) or {}

    # Apply the updates
    for key_path, value in dict_update.items():
        keys = _parse_key_path(key_path)
        _deep_update(config, keys, value)

    # Save the updated configuration back to the configuration file
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    return config

def _parse_key_path(key_path):
    """
    Parse a dot-separated key path that may contain list indices.

    Parameters
    ----------
    key_path : str
        Dot-separated key path, can contain list indices in square brackets,
        e.g., 'key1.key2[0].key3' or 'key1.key2[1]'.

    Returns
    -------
    keys : list
        List of keys and indices, e.g., ['key1', 'key2', 0, 'key3'] or ['key1', 'key2', 1]
    """
    keys = []
    parts = key_path.split('.')

    for part in parts:
        # Check if part is a key with list index, e.g., "key[0]"
        match = re.match(r'^(.+?)\[(\d+)\]$', part)
        if match:
            # If there's a key before the index, add the key first
            key = match.group(1)
            index = int(match.group(2))
            if key:  # only add if the key is not empty
                keys.append(key)
            # Add the index as integer
            keys.append(index)
        else:
            # Check if part is a pure index, e.g., "[0]"
            pure_index_match = re.match(r'^\[(\d+)\]$', part)
            if pure_index_match:
                index = pure_index_match.group(1)
                keys.append(index)
            else:
                # Regular key
                keys.append(part)

    return keys

def _deep_update(d, keys, value):
    """
    A helper function to recursively update nested dict `d` at `keys` with `value`,
    supporting list indices in the key path.

    Parameters
    ----------
    d : dict or list
        The dictionary or list to update.
    keys : list
        List of keys and indices to traverse.
    value : any
        The value to set at the final key/index.

    Raises
    ------
    `KeyError`: If a key doesn't exist in a dictionary.
    `IndexError`: If an index is out of range for a list.
    `TypeError`: If trying to index a non-list or access a non-dict.
    """
    if not keys:
        return

    current_key = keys[0]
    remaining_keys = keys[1:]

    if not remaining_keys:
        # Set the value at the final key
        if isinstance(d, dict):
            d[current_key] = value
        elif isinstance(d, list):
            if isinstance(current_key, int):
                if current_key < len(d):
                    d[current_key] = value
                else:
                    raise IndexError(f"Index {current_key} out of range for list of length {len(d)}.")
            else:
                raise TypeError(f"Cannot use string key '{current_key}' with list.")
        else:
            raise TypeError(f"Cannot set value on type {type(d)}.")

    else:
        # Traverse deeper at the current key when it's not the final one
        if isinstance(d, dict):
            if current_key in d:
                _deep_update(d[current_key], remaining_keys, value)
            else:
                raise KeyError(f"Key '{current_key}' not found in dictionary.")

        elif isinstance(d, list):
            if isinstance(current_key, int):
                if current_key < len(d):
                    _deep_update(d[current_key], remaining_keys, value)
                else:
                    raise IndexError(f"Index {current_key} out of range for list of length {len(d)}.")
            else:
                raise TypeError(f"Cannot use string key '{current_key}' with list.")

        else:
            raise TypeError(f"Cannot traverse into type {type(d)} at key '{current_key}'.")


def to_python_floats(obj):
    """Recursively convert all numeric values in `obj` to native Python floats. Booleans and non-numeric types are preserved."""
    # For dictionary-like objects (dict, OrderedDict, etc.)
    if isinstance(obj, Mapping):
        return obj.__class__((k, to_python_floats(v)) for k, v in obj.items())

    # For astropy Quantity objects, use their value
    if isinstance(obj, u.Quantity):
        return to_python_floats(obj.value)

    # For numpy arrays, convert to list and recurse
    if isinstance(obj, np.ndarray):
        return to_python_floats(obj.tolist())

    # For numpy scalar types
    if isinstance(obj, np.generic):
        if isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj.item()

    # For general iterables (list, tuple, set), but not strings/bytes/bytearray
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, bytearray)):
        # Preserve the original container type (list, tuple, set)
        return obj.__class__(to_python_floats(item) for item in obj)

    # For native Python numeric types
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        return float(obj)

    else:
        warnings.warn(f"Cannot convert {obj} of type {type(obj)} to Python floats, or all numeric values have been converted already. The object is returned as is.")
        return obj


def update_dict(dict1, dict2):
    """
    Update `dict1` with `dict2` recursively on intersection keys.
    """
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            update_dict(dict1[key], value)
        elif key in dict1:
            dict1[key] = value

    return dict1

def update_dict_none(dict1, dict2):
    """
    Update `dict1` with `dict2` only on keys where `dict1` has `None` values or is missing.
    """
    updated_keys = []
    updated_dict = {}

    # Process keys from dict2 in their original order
    for key, value in dict2.items():
        # Check if this key should be updated in dict1
        if dict1.get(key, None) is None and value is not None:
            updated_dict[key] = value
            updated_keys.append(key)
        else:
            # Use the value from dict1 if it exists
            if key in dict1:
                updated_dict[key] = dict1[key]

    # Add any remaining keys from dict1 that weren't in dict2
    for key, value in dict1.items():
        if key not in updated_dict:
            updated_dict[key] = value

    return updated_dict, updated_keys


### ------ Metadata Retrieval ------ ###
def get_source_metadata(name, sector):
    """
    Retrieve the metadata of the source without downloading any `postcard` or `TESScut`.

    Parameters
    ----------
    name : str
        The name of the source.
    sector : int
        The sector number.

    Returns
    ----------
    metadata_dict : dict
        A dictionary containing the metadata of the source, including sector, camera, CCD, CCD coordinates, TIC ID, RA, Dec, Gaia ID, TESS magnitude,
        whether it has `postcard` or `TESScut`, TIC version, and contamination ratio.
    """
    # Retrieve the coordinates of the source from its name
    warnings.filterwarnings('ignore')

    coords = coords_from_name(name)
    if coords is None:
        raise ValueError(f"Could not resolve coordinates for {name}.")

    # Retrieve TIC ID and Tmag from coordinates
    tic_info = tic_from_coords(coords)
    if tic_info is None:
        raise ValueError(f"Could not resolve TIC information for {name}.")
    tic = tic_info[0]
    tess_mag = tic_info[1][0]

    # Retrieve the Gaia ID from coordinates
    gaia = gaia_from_coords(coords)
    if gaia is None:
        raise ValueError(f"Could not resolve Gaia ID for {name}.")

    # Retrieve the location of the source in the TESS sector using tess_stars2px
    result = tess_stars2px(tic, coords[0], coords[1])
    sector_mask = result[3] == sector
    if not np.any(sector_mask):
        raise ValueError(f"No data available for {name} in sector {sector}.")

    camera = result[4][sector_mask][0]
    ccd = result[5][sector_mask][0]
    ccd_x = result[6][sector_mask][0]
    ccd_y = result[7][sector_mask][0]
    ccd_coords = (float(ccd_x), float(ccd_y))


    # Check whether the source uses postcard or TESScut by default
    # define the postcard naming format
    info_str = "{0:04d}-{1}-{2}-{3}".format(sector, camera, ccd, "cal")
    postcard_fmt = "postcard-s{0}-{{0:04d}}-{{1:04d}}"
    postcard_fmt = postcard_fmt.format(info_str)

    # retrieve the postcard centers from the postcard_centers file
    eleanorpath = os.path.dirname(os.path.abspath(__import__('eleanor').__file__))
    guide_url = eleanorpath + '/postcard_centers.txt'
    guide = Table.read(guide_url, format="ascii")
    post_args = np.where((np.abs(guide['x'].data - ccd_x) <= 100) & (np.abs(guide['y'].data - ccd_y) <= 100))
    post_cens = guide[post_args]

    # find the closest postcard
    closest_x, closest_y = np.argmin(np.abs(post_cens['x'] - ccd_x)), np.argmin(np.abs(post_cens['y'] - ccd_y))
    postcard_name = postcard_fmt.format(post_cens['x'].data[closest_x], post_cens['y'].data[closest_y])

    postcard_obs = Observations.query_criteria(provenance_name="ELEANOR", target_name=postcard_name, obs_collection="HLSP")

    if len(postcard_obs) > 0:
        postcard = True
    else:
        postcard = False


    metadata_dict = {
        'source.sector': sector,
        'source.camera': int(camera),
        'source.ccd': int(ccd),
        'source.ccd_coords': ccd_coords,
        'source.tic': int(tic),
        'source.RA': float(coords[0]),
        'source.Dec': float(coords[1]),
        'source.gaia': int(gaia),
        'source.tess_mag': float(tess_mag),
        'eleanor.source.postcard': postcard,
        'eleanor.source.tesscut': not postcard,
    }

    return metadata_dict


def epoch_time_to_btjd(epoch_time, period):
    """
    Convert epoch time from `BJD` to `BTJD`.
    """
    tess_time = 2457000

    while epoch_time < tess_time:
       epoch_time += period
    epoch_time -= tess_time

    return epoch_time




### ------ FITS File Handling ------ ###
def inspect_fits_file(fits_path, fits_inspection_dir, fits_inspection_path=None):
    """
    Inspect a `FITS` file and write its metadata into a text file.
    """
    # Define the filenames
    fits_fn = os.path.basename(fits_path)
    fits_fn_pure = os.path.splitext(fits_fn)[0]
    if fits_inspection_path is None:
        fits_inspection_path = fits_inspection_dir + f"/{fits_fn_pure}.txt"
    fits_inspection_file = open(fits_inspection_path, "w", encoding='utf-8')


    hdulist = fits.open(fits_path)

    # Print the overall FITS information to the file
    hdulist.info(output=fits_inspection_file)
    fits_inspection_file.write("\n\n")

    # Print the header of the primary HDU to the file
    primary_hdu = hdulist[0]
    primary_header = primary_hdu.header
    fits_inspection_file.write("Primary Header:\n")
    for key, value in primary_header.items():
        comment = primary_header.comments[key]
        fits_inspection_file.write(f"{key} = {value} / {comment}\n")
    fits_inspection_file.write("\n")

    hdulist.close()
    fits_inspection_file.close()


def match_fits_file(fits_path, type, config):
    """
    Check whether a `FITS` file matches the specific source, mission, sector, author and exptime.

    Parameters
    ----------
    type : str, 'tpf' & 'targetpixelfile' or 'lc' & 'lightcurve'
        The type of the `FITS` file, 'tpf' & 'targetpixelfile' for target pixel file `FITS` file, and 'lc' & 'lightcurve' for light curve `FITS` file.
    config : dict
        A dictionary containing the metadata of the source, should at least include TIC ID and Gaia ID of the source, mission, sector, author and exptime.

    Returns
    ----------
    `True` if all metadata match, `False` otherwise.
    """
    # Validate the type parameter
    if type.lower() not in ['tpf', 'targetpixelfile', 'lc', 'lightcurve']:
        raise ValueError("Invalid type parameter. Must be 'tpf' & 'targetpixelfile' (for target pixel file FITS file), or 'lc' & 'lightcurve' (for light curve FITS file).")


    # Define the source parameters and downloading criteria
    tic = config['tic']
    gaia = config['gaia']
    RA = config['RA']
    Dec = config['Dec']
    coords = (RA, Dec)
    mission = config['mission']
    sector = config['sector']
    if type.lower() in ['tpf', 'targetpixelfile']:
        author = config['author']
        exptime = config['exptime'] if config['exptime'] != 'None' else config['exptime']
        tpf_height = config['tpf_height']
        tpf_width = config['tpf_width']
    elif type.lower() in ['lc', 'lightcurve']:
        author = config['author']
        exptime = config['exptime'] if config['exptime'] != 'None' else config['exptime']


    # Read the FITS primary header
    hdulist = fits.open(fits_path)
    primary_hdu = hdulist[0]
    primary_header = primary_hdu.header


    # Extract the relevant metadata from the FITS primary header
    fits_tic = primary_header.get('TICID', None) or primary_header.get('OBJECT', None)

    # extract the coordinates of the source only when matching target pixel file FITS file obtained by TESScut
    if type.lower() in ['tpf', 'targetpixelfile'] and author.lower() == "tesscut":
        fits_ra = primary_header.get('RA_OBJ', None)
        fits_dec = primary_header.get('DEC_OBJ', None)
        fits_coords = (fits_ra, fits_dec) if fits_ra is not None and fits_dec is not None else None

    fits_mission = primary_header.get('TELESCOP', None).strip().lower() if primary_header.get('TELESCOP', None) is not None else None

    fits_sector = primary_header.get('SECTOR', None)

    fits_author = primary_header.get('ORIGIN', None).strip().lower() if primary_header.get('ORIGIN', None) is not None else None

    # try extracting the exptime from the 'TIMEDEL' keyword from the primary header first (typically exists in the CDIPS light curve FITS file); if not available, calculate it from the data
    if primary_header.get('TIMEDEL', None) is not None:
        fits_exptime = (primary_header.get('TIMEDEL', None) * u.day).to(u.second).value # convert exptime to seconds
    elif type.lower() in ['lc', 'lightcurve']:
        lc = lk.read(fits_path)
        for k in range(len(lc.time)-1):
            time_interval = lc.time[k+1] - lc.time[k]
            if time_interval > 0:
                fits_exptime = time_interval.to(u.second).value # convert exptime to seconds
                break
    elif type.lower() in ['tpf', 'targetpixelfile']:
        tpf = lk.read(fits_path)
        for k in range(len(tpf.time)-1):
            time_interval = tpf.time[k+1] - tpf.time[k]
            if time_interval > 0:
                fits_exptime = time_interval.to(u.second).value
                break

    # try extracting the TPF size (i.e., height and width) from the 'NAXIS1' and 'NAXIS2' keywords from the primary header (typically exists in the TESS target pixel file FITS file)
    if type.lower() in ['tpf', 'targetpixelfile']:
        aperture_hdu = hdulist[2]
        aperture_header = aperture_hdu.header
        fits_tpf_width = aperture_header.get('NAXIS1', None)
        fits_tpf_height = aperture_header.get('NAXIS2', None)


    hdulist.close()


    # Convert some specific metadata expressions
    if "spoc" in author.lower():
        author = "ames"
    if author.lower() == "tesscut":
        author = "mast"


    # Check if all metadata match
    if type.lower() in ['tpf', 'targetpixelfile']:
        if author == "ames":
            if (np.char.find(str(fits_tic), str(tic)) >= 0 and
                    fits_mission == mission.lower() and
                    fits_sector == sector and
                    np.char.find(str(fits_author), str(author.lower())) >= 0 and
                    abs(fits_exptime - exptime) < 0.1): # allow a tolerance of 0.1 s in exptime comparison
                if fits_tpf_height != tpf_height or fits_tpf_width != tpf_width:
                    warnings.warn(f"The TPF size does not match: expected {tpf_width}x{tpf_height}, but got {fits_tpf_width}x{fits_tpf_height}. The 'tpf_width' and 'tpf_height' parameters in the configuration file will be updated accordingly.")
                return True
        elif author == "mast":
            if (np.allclose(fits_coords, coords, atol=1/36000) and  # allow a tolerance of 0.1 arcsec in coordinate comparison
                    fits_mission == mission.lower() and
                    fits_sector == sector and
                    np.char.find(str(fits_author), str(author.lower())) >= 0 and
                    abs(fits_exptime - exptime) < 0.1 and  # allow a tolerance of 0.1 s in exptime comparison
                    fits_tpf_height == tpf_height and
                    fits_tpf_width == tpf_width):
                return True

    elif type.lower() in ['lc', 'lightcurve']:
        if (np.char.find(str(fits_tic), str(tic)) >= 0 and
            fits_mission == mission.lower() and
            fits_sector == sector and
            np.char.find(str(fits_author), str(author.lower())) >= 0 and
            abs(fits_exptime - exptime) < 0.1): # allow a tolerance of 0.1 s in exptime comparison
            return True

    else:
        return False


def search_matched_fits_file(fits_dir, type, config):
    """
    Search for a `FITS` file that matches the specific source, mission, sector, author and exptime in a directory (including sub-directories).

    Parameters
    ----------
    type : str, 'tpf' & 'targetpixelfile' or 'lc' & 'lightcurve'
        The type of the `FITS` file, 'tpf' & 'targetpixelfile' for target pixel file `FITS` file, and 'lc' & 'lightcurve' for light curve `FITS` file.
    config : dict
        A dictionary containing the metadata of the source, should at least include TIC ID and Gaia ID of the source, mission, sector, author and exptime.

    Returns
    ----------
    fits_path : str or None
        The path of the matched `FITS` file if found, `None` otherwise.
    """
    # List all FITS files in the directory (including sub-directories)
    fits_path_list = [os.path.join(root, f) for root, _, files in os.walk(fits_dir) for f in files if f.endswith('.fits')]

    # Check each FITS file for a match
    for fits_path in fits_path_list:
        if match_fits_file(fits_path, type, config):
            return fits_path

    return None


def format_fits_fn(fits_metadata_dict):
    """
    Format the standardized filename for a `FITS` file based on the metadata of the source and data.

    Parameters
    ----------
    fits_metadata_dict : dict
        A dictionary containing the metadata of the source and data, should include the following keys:
        - `type` : str, the type of the `FITS` file, 'tpf' & 'targetpixelfile' (for target pixel file `FITS` file),
                'lc' & 'lightcurve' (for light curve `FITS` file), or 'targetdata' (for `eleanor.TargetData` `FITS` file)
        - `name` : str, the name of the source
        - `mission` : str, the mission name, e.g., 'TESS'
        - `sector` : int, the sector number
        - `author` : str, the author of the data
        - `exptime` : int, the exposure time of the data in seconds
        - `tpf_height` : int, optional, the height of the target pixel file in pixels (required if type is `tpf` or `targetpixelfile`)
        - `tpf_width` : int, optional, the width of the target pixel file in pixels (required if type is `tpf` or `targetpixelfile`)
        - `aperture_mask_type` : str, optional, the type of the aperture mask used when extracting light curve via `lightkurve.TESSTargetPixelFile.extract_aperture_photometry()` method (required if author contains `lightkurve_aperture`)
        - `flux_method` : str, optional, the flux extraction method used when extracting light curve via `lightkurve.TESSTargetPixelFile.extract_aperture_photometry()` method (required if author contains `lightkurve_aperture`)
        - `centroid_method` : str, optional, the method of estimating the centroids used when extracting light curve via `lightkurve.TESSTargetPixelFile.extract_aperture_photometry()` method (required if author contains `lightkurve_aperture`)
        - `lc_type` : str, optional, the type of the `eleanor` light curve, must be one of `Raw`, `Corrected`, `PCA`, `PSF` (required if type is `lc` or `lightcurve` and author is `eleanor`)
        - `postcard` : bool, optional, whether the data is obtained from `postcard` (required if author is `eleanor`)

    Returns
    ----------
    fits_fn : str
        The standardized filename for the `FITS` file.
    """
    type = fits_metadata_dict['type']
    name = fits_metadata_dict['name']
    mission = fits_metadata_dict['mission']
    sector = fits_metadata_dict['sector']
    author = fits_metadata_dict['author']
    exptime = fits_metadata_dict['exptime']

    tpf_height = fits_metadata_dict.get('tpf_height', None)
    tpf_width = fits_metadata_dict.get('tpf_width', None)

    aperture_mask_type = fits_metadata_dict.get('aperture_mask_type', None)
    flux_method = fits_metadata_dict.get('flux_method', None)
    centroid_method = fits_metadata_dict.get('centroid_method', None)

    lc_type = fits_metadata_dict.get('lc_type', None)
    postcard = fits_metadata_dict.get('postcard', None)


    # Validate the input parameters
    if type.lower() not in ['tpf', 'targetpixelfile', 'lc', 'lightcurve', 'targetdata']:
        raise ValueError("Invalid 'type' parameter. Must be 'tpf' & 'targetpixelfile' (for target pixel file FITS file), 'lc' & 'lightcurve' (for light curve FITS file), or 'targetdata' (for eleanor.TargetData FITS file).")

    if type.lower() in ['tpf', 'targetpixelfile'] and (tpf_height is None or tpf_width is None):
        raise ValueError("Both 'tpf_height' and 'tpf_width' parameters must be provided when type is 'tpf' or 'targetpixelfile'.")

    if type.lower() == 'targetdata':
        if author.lower() != 'eleanor':
            raise ValueError("The 'author' parameter must be 'eleanor' when type is 'targetdata'.")

    if "extracted" in author.lower():
        if type.lower() not in ['lc', 'lightcurve']:
            raise ValueError("The 'type' parameter must be 'lc' or 'lightcurve' when the 'author' parameter contains 'extracted' (i.e., the data is a lightcurve extracted from TPF).")
        if "lightkurve_aperture" in author.lower():
            if tpf_height is None or tpf_width is None or aperture_mask_type is None or flux_method is None or centroid_method is None:
                raise ValueError("The 'tpf_height', 'tpf_width', 'aperture_mask_type', 'flux_method', and 'centroid_method' parameters must be provided when the 'author' parameter contains 'lightkurve_aperture'\n"
                                 "(i.e., the lightcurve is extracted via lightkurve.TESSTargetPixelFile.extract_aperture_photometry() method).")

    if author.lower() == "eleanor":
        if postcard is None:
            raise ValueError("'postcard' parameter must be provided (True or False) when author is 'eleanor'.")
        if type.lower() in ['lc', 'lightcurve']:
            if lc_type is None:
                raise ValueError("'lc_type' parameter must be provided (and must be one of ['Raw', 'Corrected', 'PCA', 'PSF']) when type is 'lc' or 'lightcurve' and author is 'eleanor'.")
            elif lc_type.lower() not in ['raw', 'corrected', 'pca', 'psf']:
                raise ValueError("Invalid 'lc_type' parameter. Must be one of ['Raw', 'Corrected', 'PCA', 'PSF'].")


    # Format the standardized FITS filename
    if type.lower() in ['tpf', 'targetpixelfile']:
        fits_fn_pure = f"{name}_{mission}_Sector-{sector}_Author-{author}_{tpf_width}x{tpf_height}_Exptime={exptime}s_TPF"
    elif type.lower() in ['lc', 'lightcurve']:
        if author.lower() == "eleanor":
            if postcard:
                fits_fn_pure = f"{name}_{mission}_Sector-{sector}_Author-Eleanor_Exptime={exptime}s_{lc_type}_LC_Postcard"
            else:
                fits_fn_pure = f"{name}_{mission}_Sector-{sector}_Author-Eleanor_Exptime={exptime}s_{lc_type}_LC_TESScut"
        elif "extracted" in author.lower():
            if "lightkurve_aperture" in author.lower():
                fits_fn_pure = f"{name}_{mission}_Sector-{sector}_Author-{author}_Exptime={exptime}s_{tpf_width}x{tpf_height}_Aperture-{aperture_mask_type}_Flux-{flux_method}_Centroid-{centroid_method}_LC"
        else:
            fits_fn_pure = f"{name}_{mission}_Sector-{sector}_Author-{author}_Exptime={exptime}s_LC"
    elif type.lower() == 'targetdata':
        if postcard:
            fits_fn_pure = f"{name}_{mission}_Sector-{sector}_Author-Eleanor_Exptime={exptime}s_TargetData_Postcard"
        else:
            fits_fn_pure = f"{name}_{mission}_Sector-{sector}_Author-Eleanor_Exptime={exptime}s_TargetData_TESScut"

    fits_fn_ext = ".fits"
    fits_fn = fits_fn_pure + fits_fn_ext

    return fits_fn


def format_lc_fits_fn_by_provenance(provenance, config):
    """
    Format the standardized filename for a light curve `FITS` file based on its provenance.
    The metadata of the light curve will be collected automatically according to its provenance.

    Parameters
    ----------
    provenance : str, 'extracted', 'downloaded', 'eleanor' or 'lightkurve'
        The provenance of the light curve.
        `extracted` for light curve extracted from TPF in `B_ab_Light_Curve_Extracting_From_TPF`;
        `downloaded` for light curve downloaded in `B_bb_Light_Curve_Downloading`;
        `eleanor` for light curve extracted and preliminarily processed via `eleanor` in `B_c_TESS_Light_Curve_Extracting_and_Preliminary_Processing_Eleanor`;
        `lightkurve` for light curve further processed via `Lightkurve` in `C_Light_Curve_Further_Processing_Lightkurve`.
    config : dict
        The configurations dictionary.

    Returns
    ----------
    lc_fn : str
        The standardized filename for the light curve `FITS` file.
    """
    name = config['source']['name']
    mission = config['mission']['mission']
    sector = config['source']['sector']
    exptime = config['source']['exptime']

    lc_fn = None


    if provenance == 'extracted':
        tpf_author = config['tpf_selecting_criteria']['author']
        exptime = config['tpf_selecting_criteria']['exptime'] if config['tpf_selecting_criteria']['exptime'] is not None else exptime
        tpf_height = config['tpf_selecting_criteria']['tpf_height']
        tpf_width = config['tpf_selecting_criteria']['tpf_width']
        lightcurve_extracting_method = config['lightcurve_extracting']['method']
        author = f"{tpf_author}-{lightcurve_extracting_method}-Extracted"

        if lightcurve_extracting_method == 'lightkurve_aperture':
            aperture_mask_type = config['lightcurve_extracting']['lightkurve_aperture']['aperture_mask_type']
            threshold = config['lightcurve_extracting']['lightkurve_aperture']['threshold']
            flux_method = config['lightcurve_extracting']['lightkurve_aperture']['flux_method']
            centroid_method = config['lightcurve_extracting']['lightkurve_aperture']['centroid_method']
            lc_fits_metadata_dict = {'type': 'lc', 'name': name, 'mission': mission, 'sector': sector, 'author': author, 'exptime': exptime, 'tpf_height': tpf_height, 'tpf_width': tpf_width, 'aperture_mask_type': aperture_mask_type, 'flux_method': flux_method, 'centroid_method': centroid_method}
            lc_fn = format_fits_fn(lc_fits_metadata_dict)

    elif provenance == 'downloaded':
        author = config['lightcurve_downloading_criteria']['author']
        exptime = config['lightcurve_downloading_criteria']['exptime'] if config['lightcurve_downloading_criteria']['exptime'] is not None else exptime
        lc_fits_metadata_dict = {'type': 'lc', 'name': name, 'mission': mission, 'sector': sector, 'author': author, 'exptime': exptime}
        lc_fn = format_fits_fn(lc_fits_metadata_dict)

    elif provenance == 'eleanor':
        author = "Eleanor"
        lc_type = config['eleanor']['lc_type']
        postcard = config['eleanor']['source']['postcard']
        lc_fits_metadata_dict = {'type':'lc', 'name': name, 'mission': mission, 'sector': sector, 'author': author, 'exptime': exptime, 'lc_type': lc_type, 'postcard': postcard}
        lc_fn = format_fits_fn(lc_fits_metadata_dict)

    elif provenance == 'lightkurve':
        provenance = config['lightkurve']['lightcurve_provenance']
        if provenance == "lightkurve":
            raise ValueError("The 'lightcurve_provenance' parameter can't be set to 'lightkurve' in C_Light_Curve_Further_Processing_Lightkurve (i.e., in the key 'lightcurve_provenance' of the 'lightkurve' section in the configuration file).\n"
                             "Please set it to 'extracted', 'downloaded' or 'eleanor'.")
        correction = config['lightkurve']['correction']
        lc_raw_fn = format_lc_fits_fn_by_provenance(provenance, config)
        lc_fn = lc_raw_fn.replace("_LC", f"{correction}_LC", -1)


    return lc_fn


def update_fits_headers(fits_file, dict_headers_update, open=False):
    """
    Update header information for multiple `HDU`s in a `FITS` file simultaneously.
    Add, modify, or delete keywords in the headers of specified `HDU`s.

    Parameters
    ----------
    fits_file : str or `astropy.io.fits.HDUList`
        Path to the `FITS` file or an `HDUList` object to be updated.
    dict_headers_update : dict
        Dictionary where keys are `HDU` indices and values are dictionaries of header updates for that `HDU`.
        Format for header updates:
        - Add/Modify: {'KEYWORD': value} or {'KEYWORD': (value, comment)}
        - Delete: {'KEYWORD': `None`} or {'KEYWORD': (`None`, "comment")}
    open : bool, optional
        If `True`, the function will return the `HDUList` object without closing it. Default is `False`.

    Returns
    ----------
    hdulist : `astropy.io.fits.HDUList` or `None`
        The updated `HDUList` object if `open` is `True`, otherwise `None`.
    """
    # Validate the FITS input
    # open the FITS file if a file path is provided
    if isinstance(fits_file, str):
        hdulist = fits.open(fits_file, mode='update')
    # use the provided HDUList object directly if the FITS file is already opened as an HDUList object
    elif isinstance(fits_file, fits.HDUList):
        hdulist = fits_file
    else:
        raise ValueError(f"fits must be a file path or HDUList object. Got {type(fits_file)} instead.")


    # Update header for each HDU
    for hdu_index, dict_header_update in dict_headers_update.items():
        # Validate HDU index
        if hdu_index >= len(hdulist):
            raise IndexError(f"HDU index {hdu_index} out of range. The FITS file has {len(hdulist)} HDUs in total.")

        header = hdulist[hdu_index].header

        for keyword, value in dict_header_update.items():
            # Handle delete operations
            if value is None or (isinstance(value, tuple) and value[0] is None):
                if keyword.upper() in header:
                    del header[keyword.upper()]
                continue
            # Handle values with comments
            if isinstance(value, tuple) and len(value) == 2:
                value, comment = value
                header.set(keyword.upper(), value, comment)
            else:
                header.set(keyword.upper(), value)

    # Save all changes to the file if a FITS file path is provided and opened
    if isinstance(fits_file, str):
        hdulist.flush()


    if open:
        return hdulist
    else:
        hdulist.close()
        return None




### ------ Data Metrics ------ ###
def running_mean(data, window_size):
    """
    Returns the running mean of a data array.

    Parameters
    ----------
    data : array-like
        The running mean will be computed on this data.
    window_size : int
        Window size used to compute the running mean.

    Returns
    -------
    running_mean : array-like
        The running mean of the input data array.
    """
    # Validate the window size
    if window_size > len(data):
        window_size = len(data)

    cumsum = np.cumsum(np.insert(data, 0, 0))
    running_mean = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

    return running_mean


def clipped_mean(data, sigma=3.0):
    """
    Calculate the sigma-clipped mean of a data series.
    """
    data = np.asarray(data)
    mean = np.nanmean(data)
    std = np.nanstd(data)
    data_clipped = data[np.abs(data - mean) < sigma * std]

    return np.nanmean(data_clipped)


def max_deviation(arr):
    """
    Calculate the deviation of the most deviated data point in terms of standard deviations
    from the NaN-median of the other data points in a data series.
    """
    array = np.asarray(arr)
    median_all = np.nanmedian(array)
    abs_deviation = np.abs(array - median_all)
    max_dev_index = np.nanargmax(abs_deviation)
    max_dev_value = array[max_dev_index]

    array_clipped = np.delete(array, max_dev_index)
    median_clipped = np.nanmedian(array_clipped)
    std_clipped = np.nanstd(array_clipped)

    max_dev_n_stds = np.abs(max_dev_value - median_clipped) / std_clipped if std_clipped != 0 else float('inf')

    return max_dev_index, max_dev_value, max_dev_n_stds


def format_value_with_uncertainty(param, precision=8):
    """
    Format a parameter value with its uncertainties to a specified precision.

    Parameters
    ----------
    param : float or tuple/list/`numpy.ndarray` of three values
        The parameter value. It can be a single value or a tuple/list/array of three values: value, lower uncertainty, upper uncertainty.
    precision : int, optional
        The number of decimal places to round the value and uncertainties to. Default is `8`.

    Returns
    ----------
    formatted_param : float or list of three values
        The formatted parameter value.
        If `param` is a single value, returns the value rounded to the specified `precision`.
        If `param` is a tuple/list/`numpy.ndarray` of three values (value, lower uncertainty, upper uncertainty),
        returns a list of the value and uncertainties rounded to the specified `precision`.
    """
    if isinstance(param, (tuple, list, np.ndarray)) and len(param) == 3:
        val, lower, upper = param
        if isinstance(val, (tuple, list, np.ndarray)) and isinstance(lower, (tuple, list, np.ndarray)) and isinstance(upper, (tuple, list, np.ndarray)):
            formatted_param = [[round(v, precision) for v in val], [-round(l, precision) for l in lower], [+round(u, precision) for u in upper]]
        else:
            formatted_param = [round(val, precision), -round(lower, precision), +round(upper, precision)]
    else:
        if isinstance(param, (tuple, list, np.ndarray)):
            formatted_param = [round(p, precision) for p in param]
        else:
            formatted_param = round(param, precision)

    return formatted_param




### ------ TPF Processing ------ ###
def aperture_overlay(data, aperture_mask, data_type='Flux', cadence=0, ax=None,
                     aperture_color='red', hatch_density=5, show_colorbar=True, **kwargs):
    """
    Overlay an aperture mask on TPF with hatch pattern appearance.

    Parameters
    ----------
    data : array
        3D array with shape `(time, nrows, ncols)` representing the data (e.g., `flux`, `flux_err`, etc.) of the TPF.
    aperture_mask : array
        2D array with the same `(nrows, ncols)` as `data` or 3D array with the same shape as `data`, with values in range [0,1] representing the aperture mask weights.
    ax : `matplotlib.axes.Axes`, optional
        The matplotlib Axes object to plot on. If no Axes is provided, a new one will be generated.
    data_type : str, optional
        Type of data to plot, e.g., 'Flux', 'Flux_err', etc. Will be displayed in the colorbar label. Default is `'Flux'`.
    cadence : int, optional
        Cadence index to display. Default is `0` (i.e., the first cadence).
    aperture_color : str, optional
        Color name for the aperture overlay. Default is `'red'`.
    hatch_density : int, optional
        Density of hatch pattern (number of lines). Default is `5`.
    show_colorbar : bool, optional
        Whether to show the colorbar. Default is `True`.
    kwargs : dict
        Dictionary of additional arguments to be passed to TPF image plotting, i.e., `matplotlib.axes.Axes.imshow()`.

    Returns
    -------
    ax : `matplotlib.axes.Axes`
        The matplotlib `Axes` object with the plot.
    """
    # Validate the input parameters
    if data.ndim != 3:
        raise ValueError("'data' must be a 3D array with shape (time, nrows, ncols).")
    if aperture_mask.ndim == 3 and aperture_mask.shape != data.shape:
        raise ValueError("If 'aperture_mask' is a 3D array, it must have same shape as 'data'.")
    elif aperture_mask.ndim == 2 and aperture_mask.shape != data.shape[1:]:
        raise ValueError("If 'aperture_mask' is a 2D array, it must have same (nrows, ncols) as 'data'.")
    elif aperture_mask.ndim not in [2, 3]:
        raise ValueError("'aperture_mask' must be a 2D array with the same (nrows, ncols) as data or 3D array with the same shape as data.")
    if cadence < 0 or cadence >= data.shape[0]:
        raise ValueError(f"'cadence' must be in range [0, {data.shape[0] - 1}].")

    # Use the aperture mask of the specified cadence if it is a 3D array
    if aperture_mask.ndim == 3:
        aperture_mask = aperture_mask[cadence]


    # Create a new Axes if not provided
    if ax is None:
        fig, ax = plt.subplots()

    # Plot data image
    img_extent = (-0.5, data[cadence].shape[1] - 0.5, -0.5, data[cadence].shape[0] - 0.5)
    im = ax.imshow(data[cadence], origin='lower', extent=img_extent, **kwargs)

    # Set title
    ax.set_title(f"TPF (Cadence {cadence})")

    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f"{data_type} ($e^{{-}}s^{{-1}}$)")


    # Add aperture overlay using patches with hatch pattern
    # create hatch pattern for each pixel with non-zero aperture weight
    base_color = to_rgba(aperture_color)
    for i in range(aperture_mask.shape[0]):
        for j in range(aperture_mask.shape[1]):
            aperture_weight = aperture_mask[i, j]
            if aperture_weight > 0:
                # create semi-transparent color (in RGBA) based on aperture weight
                color = (base_color[0], base_color[1], base_color[2], aperture_weight) # scale alpha by aperture weight
                # create rectangle patch with hatch pattern
                aperture_patch = patches.Rectangle(xy=(j - 0.5, i - 0.5), width=1, height=1, fill=False, edgecolor=color, hatch='/' * hatch_density, linewidth=0.5, alpha=aperture_weight)
                ax.add_patch(aperture_patch)

    ax.grid(False)
    ax.set_xlabel('Pixel Column Number')
    ax.set_ylabel('Pixel Row Number')


    return ax




### ------ Lightcurve Processing ------ ###
def calculate_cdpp(lc, exptime=None, cdpp_transit_duration=6.5):
    """
    Calculate the Combined Differential Photometric Precision (`CDPP`) noise metric
    directly from the raw light curve, without applying flattening or outliers removal.

    This method computes the `CDPP` by:
    1. Normalizing the flux to parts-per-million (`ppm`);
    2. Applying a running mean with a window size equal to the transit duration;
    3. Calculating the standard deviation of the running mean series.

    Parameters
    ----------
    lc : `lightkurve.LightCurve`
        The `LightCurve` object.
    exptime : float or None, optional
        The exposure time of the light curve in units of `seconds`.
        If not provided, will be calculated from the time array of the light curve.
    cdpp_transit_duration : float, optional
        The transit duration in units of `hours` to calculate `CDPP`. Default is `6.5`.

    Returns
    -------
    cdpp : float
        The `CDPP` noise metric in parts-per-million (`ppm`).

    Notes
    -----
    This method differs from `lightkurve.LightCurve.estimate_cdpp()` in that it:
    - Does NOT apply `Savitzky-Golay` filtering (flattening);
    - Does NOT perform sigma-clipping to remove outliers;
    - Works directly on the input light curve flux values.
    """
    # Calculate the exposure time from the time array of the light curve if not provided
    if exptime is None:
        exptime_list = []
        for k in range(len(lc.time.value)-1):
            exptime_single = lc.time.value[k+1] - lc.time.value[k]
            exptime_list.append(exptime_single)
        exptime = clipped_mean(exptime_list)

    cleaned_lc = lc.remove_nans()

    normalized_lc = cleaned_lc.normalize("ppm")

    mean = running_mean(data=normalized_lc.flux, window_size=int(cdpp_transit_duration * 3600 / exptime))
    cdpp = float(np.std(mean)) * 1e6 # convert the cdpp value to unit of ppm

    return cdpp


def sort_lc(lc):
    """
    Sort a `lightkurve.LightCurve` object by time in ascending order.
    """
    time = lc.time
    flux = lc.flux
    flux_err = lc.flux_err

    if np.all(np.diff(time.value) > 0):
        return lc

    else:
        # Retrieve the indices that would sort the time array
        sort_indices = np.argsort(time.value)

        sorted_time = time[sort_indices]
        sorted_flux = flux[sort_indices]
        sorted_flux_err = flux_err[sort_indices]

        sorted_lc = lc.copy()
        sorted_lc.time = sorted_time
        sorted_lc.flux = sorted_flux
        sorted_lc.flux_err = sorted_flux_err

        return sorted_lc




### ------ Transit Fitting ------ ###
### General ###
def _parse_transit_model(transit_model_name):
    """
    Parse the transit model name and return the corresponding `PyTransit` transit model instance.

    Parameters
    ----------
    transit_model_name : str
        Name of the `PyTransit` transit model to use. Currently supported models are: `Quadratic`, `RoadRunner_Quadratic`.

    Returns
    -------
    transit_model : `PyTransit.TransitModel`
        A `PyTransit` transit model instance corresponding to the provided `transit_model_name`.
    """
    if transit_model_name.lower() == 'quadratic':
        transit_model = QuadraticModel()
    elif transit_model_name.lower() == 'roadrunner_quadratic':
        transit_model = RoadRunnerModel('quadratic')
    else:
        raise ValueError(f"Unsupported transit model: {transit_model_name}. Currently supported models are: 'Quadratic', 'RoadRunner_Quadratic'.")

    return transit_model


### Multiple-transit Fitting Model ###
def model_multi(params, transit_model, time):
    """
    Return the multiple-transit model light curve `flux` values based on the provided `params` and `transit_model`.

    Parameters
    ----------
    params : list or array-like
        A list or array containing the transit parameters in the following order:
        [`k`, `t0`, `p`, `a`, `i`, `ldc1`, `ldc2`],
        where:
        - `k`: normalized planetary radius, i.e., :math:`R_{\mathrm{p}}/R_{\mathrm{s}}`
        - `t0`: epoch time in `BTJD`
        - `p`: orbital period in `days`
        - `a`: normalized semi-major axis, i.e., a/:math:`R_{\mathrm{s}}`
        - `i`: orbital inclination in `radians`
        - `ldc1`: linear limb darkening coefficient
        - `ldc2`: quadratic limb darkening coefficient
    transit_model : `PyTransit.TransitModel`
        A `PyTransit` transit model instance (e.g., `QuadraticModel`).
    time : array-like or `astropy.units.Quantity`
        An array of `time` values at which to evaluate the model.

    Returns
    -------
    model_flux : array-like
        The modeled `flux` values corresponding to the input `time` array based on the provided `params` and `transit_model`.
    """
    k, t0, p, a, i, ldc1, ldc2 = params
    tm = transit_model

    if isinstance(time, np.ndarray):
        tm.set_data(time=time)
    elif isinstance(time, u.Quantity):
        tm.set_data(time=time.value)

    model_flux = tm.evaluate(k=k, t0=t0, p=p, a=a, i=i, e=0.0, w=0.0, ldc=[ldc1, ldc2])

    return model_flux


def log_likelihood_multi(params, transit_model, lc):
    """
    Calculate the `log-likelihood` of the model given the parameters and the light curve data.
    """
    model_flux = model_multi(params, transit_model, lc.time)
    model_log_likelihood = -0.5 * np.sum((lc.flux.value - model_flux)**2 / lc.flux_err.value**2)

    return model_log_likelihood

def log_prior_multi(params, lc):
    """
    Calculate the `log-prior` of the model parameters.
    """
    k, t0, p, a, i, ldc1, ldc2 = params
    if (0 < k < 1 and np.min(lc.time.value) < t0 < np.max(lc.time.value) and p > 0 and a > 1 and -np.pi/2 < i < np.pi/2 and 0 < ldc1 < 1 and 0 < ldc2 < 1):
        model_log_prior = 0.0
    else:
        model_log_prior = -float('inf')

    return model_log_prior

def log_probability_multi(params, transit_model, lc):
    """
    Calculate the `log-probability` of the model given the parameters and the light curve data.
    """
    model_log_likelihood = log_likelihood_multi(params, transit_model, lc)
    model_log_prior = log_prior_multi(params, lc)
    model_log_probability = model_log_likelihood + model_log_prior

    if np.isfinite(model_log_prior):
        return model_log_probability
    else:
        return -float('inf')


### Single-transit Fitting Model ###
def model_single(params, p, transit_model, time):
    """
    Return the single-transit model light curve `flux` values based on the provided `params` and `transit_model`.

    Parameters
    ----------
    params : list or array-like
        A list or array containing the transit parameters in the following order:
        [`k`, `t0`, `a`, `i`, `ldc1`, `ldc2`],
        where:
        - `k`: normalized planetary radius, i.e., :math:`R_{\mathrm{p}}/R_{\mathrm{s}}`
        - `t0`: epoch time in `BTJD`
        - `a`: normalized semi-major axis, i.e., a/:math:`R_{\mathrm{s}}`
        - `i`: orbital inclination in `radians`
        - `ldc1`: linear limb darkening coefficient
        - `ldc2`: quadratic limb darkening coefficient
    p : float
        The orbital period of the planet in `days`. Fixed during single-transit fitting.
    transit_model : `PyTransit.TransitModel`
        A `PyTransit` transit model instance (e.g., `QuadraticModel`).
    time : array-like or `astropy.units.Quantity`
        An array of `time` values at which to evaluate the model.

    Returns
    -------
    model_flux : array-like
        The modeled `flux` values corresponding to the input `time` array based on the provided `params` and `transit_model`.
    """
    k, t0, a, i, ldc1, ldc2 = params
    tm = transit_model

    if isinstance(time, np.ndarray):
        tm.set_data(time=time)
    elif isinstance(time, u.Quantity):
        tm.set_data(time=time.value)

    model_flux = tm.evaluate(k=k, t0=t0, p=p, a=a, i=i, e=0.0, w=0.0, ldc=[ldc1, ldc2])

    return model_flux


def log_likelihood_single(params, p, transit_model, lc):
    """
    Calculate the `log-likelihood` of the model given the parameters and the light curve data.
    """
    model_flux = model_single(params, p, transit_model, lc.time)
    model_log_likelihood = -0.5 * np.sum((lc.flux.value - model_flux)**2 / lc.flux_err.value**2)

    return model_log_likelihood

def log_prior_single(params, p, lc):
    """
    Calculate the `log-prior` of the model parameters.
    """
    k, t0, a, i, ldc1, ldc2 = params
    if (0 < k < 1 and np.min(lc.time.value) < t0 < np.max(lc.time.value) and p > 0 and a > 1 and -np.pi/2 < i < np.pi/2 and 0 < ldc1 < 1 and 0 < ldc2 < 1):
        model_log_prior = 0.0
    else:
        model_log_prior = -float('inf')

    return model_log_prior

def log_probability_single(params, p, transit_model, lc):
    """
    Calculate the `log-probability` of the model given the parameters and the light curve data.
    """
    model_log_likelihood = log_likelihood_single(params, p, transit_model, lc)
    model_log_prior = log_prior_single(params, p, lc)
    model_log_probability = model_log_likelihood + model_log_prior

    if np.isfinite(model_log_prior):
        return model_log_probability
    else:
        return -float('inf')


### Transit Properties Calculation ###
def calculate_transit_duration(k, p, a, i):
    """
    Function to calculate the transit duration (`t_14`) based on the normalized planetary radius (`k`),
    orbital period (`p`), normalized semi-major axis (`a`), and orbital inclination (`i`).
    """
    b = a * np.cos(i)
    t_14 = np.arcsin(np.sqrt((1 + k) ** 2 - b ** 2) / (a * np.sin(i))) * p / np.pi

    return t_14


def calculate_transit_depth(k, a, i, ldc, transit_model_name):
    """
    Function to calculate the transit depth based on the normalized planetary radius (`k`), normalized semi-major axis (`a`),
    orbital inclination (`i`), limb darkening coefficients (`ldc`) and the transit model.
    """
    tm = _parse_transit_model(transit_model_name)

    t0 = 0.0 # t0 should be set to 0.0
    p = 1.0 # p can be set randomly

    tm.set_data(time=[0.0]) # calculate the transit depth at t=0.0 (i.e., at the epoch time)

    flux_out = tm.evaluate(k=0, t0=t0, p=p, a=a, i=i, ldc=ldc) # calculate the out-of-transit flux at the epoch time, where k=0 means no planet
    flux_in = tm.evaluate(k=k, t0=t0, p=p, a=a, i=i, ldc=ldc) # calculate the in-transit flux at the epoch time
    transit_depth = flux_out - flux_in # calculate the transit depth

    return transit_depth


### Transit Fitting Runner ###
def run_transit_fitting(lc, transit_model_name, fit_type, params_initial=None,
                        n_walkers=32, n_steps=5000, chain_discard_proportion=0.2, chain_thin=10,
                        max_iter=3, sigma=3.0, transit_index=None):
    """
    Run `MCMC` transit fitting on a light curve using the specified transit model in the specified fit type with outliers removal iterations.

    Parameters
    -----------
    lc : `lightkurve.LightCurve`
        The `LightCurve` object. Must contain `flux` and `flux_err` columns.
    transit_model_name : str
        Name of the `PyTransit` transit model to use. Currently supported models are: `Quadratic`, `RoadRunner_Quadratic`.
    fit_type : str
        Type of fit, should be `global`, `individual` or `folded`.
    params_initial : dict, optional
        Dictionary of initial fitted transit parameters, should contain keys: `k`, `t0`, `p`, `a`, `i`, `ldc1`, `ldc2`.
        If not provided or invalid, default initial parameters will be used.
    n_walkers : int, optional
        Number of `MCMC` walkers. Default is `32`.
    n_steps : int, optional
        Number of `MCMC` steps. Default is `5000`.
    chain_discard_proportion : float, optional
        Proportion of `MCMC` chain to discard as burn-in. Default is `0.2`.
    chain_thin : int, optional
        Thinning factor of the sample chain when visualizing the process and result. Default is `10`.
    max_iter : int, optional
        Maximum number of outliers removal iterations. Default is `3`.
    sigma : float, optional
        `Sigma` threshold for outliers removal. Default is `3.0`.
    transit_index : int, optional
        Index of the transit to be fitted for individual transit fitting.

    Returns
    --------
    results : dict
        A dictionary containing the `MCMC` fitting results, including best fitted transit parameters and their uncertainties, best fitted model and residuals, and goodness-of-fit metrics.
    """
    # Validate and parse transit_model_name
    tm = _parse_transit_model(transit_model_name)

    # Validate and parse fit_type
    if fit_type.lower() == 'global':
        fit_range = 'multi'
        n_dim = 7
        params_name = ['k', 't0', 'p', 'a', 'i', 'ldc1', 'ldc2']
        log_probability_func = log_probability_multi
        model_func = model_multi
    elif fit_type.lower() in ['individual', 'folded']:
        fit_range = 'single'
        n_dim = 6
        params_name = ['k', 't0', 'a', 'i', 'ldc1', 'ldc2']
        log_probability_func = log_probability_single
        model_func = model_single
    else:
        raise ValueError(f"'fit_type' must be 'global', 'individual' or 'folded'. Got '{fit_type}' instead.")

    if fit_type.lower() != 'individual' and transit_index is not None:
        warnings.warn(f"'transit_index' is only used for individual transit fitting, so it will be ignored for {fit_type.lower()} transit fitting.")
        transit_index = None
    if fit_type.lower() == 'individual' and transit_index is None:
        raise ValueError(f"'transit_index' must be provided for individual transit fitting.")

    # Validate and set initial fitted transit parameters
    params_initial_default = {'k': 0.1, 't0': 3000.0, 'p': 4.0, 'a': 10.0, 'i': np.pi/2, 'ldc1': 0.2, 'ldc2': 0.3}

    if params_initial is None:
        params_initial = params_initial_default

    unexpected_keys = set(params_initial.keys()) - set(params_initial_default.keys())
    if unexpected_keys:
        warnings.warn(f"'params_initial' contains unexpected keys: {sorted(list(unexpected_keys))}. These keys will be ignored.")
        for key in unexpected_keys:
            params_initial.pop(key)

    params_initial, updated_keys = update_dict_none(params_initial, params_initial_default)
    if updated_keys:
        updated_map = {k: params_initial_default[k] for k in sorted(updated_keys)}
        warnings.warn(f"'params_initial' lacks or contains `None` values for keys: {sorted(list(updated_keys))}. The default initial parameters will be used for these keys: {updated_map}.")

    if fit_range.lower() == 'multi':
        params_initial_array = np.array([params_initial[key] for key in params_name])
    elif fit_range.lower() == 'single':
        params_initial_array = np.array([params_initial[key] for key in params_name if key != 'p']) # period is fixed during single-transit fitting


    # Initialize best fitted transit parameters and parameters errors dictionaries
    params_best = {'k': None, 't0': None, 'p': None, 'a': None, 'i': None, 'i_in_degree': None, 'ldc1': None, 'ldc2': None, 'ldc': None,
                   'transit_duration': None, 'transit_depth': None}
    params_best_lower_error = {'k': None, 't0': None, 'p': None, 'a': None, 'i': None, 'i_in_degree': None, 'ldc1': None, 'ldc2': None, 'ldc': None,
                               'transit_duration': None, 'transit_depth': None}
    params_best_upper_error = {'k': None, 't0': None, 'p': None, 'a': None, 'i': None, 'i_in_degree': None, 'ldc1': None, 'ldc2': None, 'ldc': None,
                               'transit_duration': None, 'transit_depth': None}

    # Initialize results dictionary
    results = {
        'n_iterations': 0,
        'n_dimensions': n_dim, 'params_name': params_name, 'params_samples': None, 'params_samples_thinned_unflattened': None,
        'params_best': params_best, 'params_best_lower_error': params_best_lower_error, 'params_best_upper_error': params_best_upper_error,
        'lc': None, 'lc_fitted': None, 'lc_residual': None,
        'residual_std': None, 'chi_square': None, 'reduced_chi_square': None,
    }


    # Initialize the transit model
    tm.set_data(lc.time.value)


    for n_iter in range(1, max_iter + 1):
        # Initialize MCMC walkers
        params_position = params_initial_array + 1e-4 * np.random.randn(n_walkers, n_dim)

        # Set up sampler arguments
        if fit_range == 'multi':
            sampler_args = (tm, lc)
        elif fit_range == 'single':
            p = params_initial['p'] # period should be fixed for single-transit fitting
            sampler_args = (p, tm, lc)

        # Set up progress bar description
        progress_desc = f"{fit_type.capitalize()} Fitting"
        if transit_index is not None:
            progress_desc += f" for Transit {transit_index:02}"
        progress_desc += f" Iteration {n_iter}: "

        # Initialize and run the MCMC sampler
        params_sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability_func, args=sampler_args)
        params_sampler.run_mcmc(params_position, n_steps, progress=True, progress_kwargs={'desc': progress_desc})

        # Retrieve the best fitted transit parameters and their uncertainties from the MCMC sampler
        params_samples = params_sampler.get_chain(discard=int(n_steps * chain_discard_proportion), flat=True)
        params_best_array = np.median(params_samples, axis=0)
        params_best_lower_error_array = (params_best_array - np.percentile(params_samples, 16, axis=0))
        params_best_upper_error_array = (np.percentile(params_samples, 84, axis=0) - params_best_array)

        if fit_range == 'multi':
            k_samples, t0_samples, p_samples, a_samples, i_samples, ldc1_samples, ldc2_samples = params_samples.T
            k_best, t0_best, p_best, a_best, i_best, ldc1_best, ldc2_best = to_python_floats(params_best_array)
            k_best_lower_error, t0_best_lower_error, p_best_lower_error, a_best_lower_error, i_best_lower_error, ldc1_best_lower_error, ldc2_best_lower_error = to_python_floats(
                params_best_lower_error_array)
            k_best_upper_error, t0_best_upper_error, p_best_upper_error, a_best_upper_error, i_best_upper_error, ldc1_best_upper_error, ldc2_best_upper_error = to_python_floats(
                params_best_upper_error_array)
        elif fit_range == 'single':
            k_samples, t0_samples, a_samples, i_samples, ldc1_samples, ldc2_samples = params_samples.T
            k_best, t0_best, a_best, i_best, ldc1_best, ldc2_best = to_python_floats(params_best_array)
            k_best_lower_error, t0_best_lower_error, a_best_lower_error, i_best_lower_error, ldc1_best_lower_error, ldc2_best_lower_error = to_python_floats(
                params_best_lower_error_array)
            k_best_upper_error, t0_best_upper_error, a_best_upper_error, i_best_upper_error, ldc1_best_upper_error, ldc2_best_upper_error = to_python_floats(
                params_best_upper_error_array)
            p_best, p_best_lower_error, p_best_upper_error = p, 0.0, 0.0

        i_in_degree_best, i_in_degree_best_lower_error, i_in_degree_best_upper_error = to_python_floats(
            [np.rad2deg(i_best), np.rad2deg(i_best_lower_error), np.rad2deg(i_best_upper_error)])

        ldc_samples = np.column_stack((ldc1_samples, ldc2_samples))
        ldc_best, ldc_best_lower_error, ldc_best_upper_error = to_python_floats(
            [[ldc1_best, ldc2_best], [ldc1_best_lower_error, ldc2_best_lower_error],
             [ldc1_best_upper_error, ldc2_best_upper_error]])

        if fit_range == 'multi':
            transit_duration_samples = [calculate_transit_duration(k, p, a, i) for k, p, a, i in zip(k_samples, p_samples, a_samples, i_samples)]
        elif fit_range == 'single':
            transit_duration_samples = [calculate_transit_duration(k, p_best, a, i) for k, a, i in zip(k_samples, a_samples, i_samples)]
        transit_duration_best = float(np.median(transit_duration_samples, axis=0))
        transit_duration_best_lower_error = float(transit_duration_best - np.percentile(transit_duration_samples, 16, axis=0))
        transit_duration_best_upper_error = float(np.percentile(transit_duration_samples, 84, axis=0) - transit_duration_best)

        transit_depth_samples = [calculate_transit_depth(k, a, i, ldc, transit_model_name) for k, a, i, ldc in zip(k_samples, a_samples, i_samples, ldc_samples)]
        transit_depth_best = float(np.median(transit_depth_samples, axis=0))
        transit_depth_best_lower_error = float(transit_depth_best - np.percentile(transit_depth_samples, 16, axis=0))
        transit_depth_best_upper_error = float(np.percentile(transit_depth_samples, 84, axis=0) - transit_depth_best)

        # Update the best fitted transit parameters and their uncertainties dictionaries
        params_best.update({'k': k_best, 't0': t0_best, 'p': p_best, 'a': a_best, 'i': i_best, 'i_in_degree': i_in_degree_best, 'ldc1': ldc1_best, 'ldc2': ldc2_best, 'ldc': ldc_best,
                            'transit_duration': transit_duration_best, 'transit_depth': transit_depth_best})
        params_best_lower_error.update({'k': k_best_lower_error, 't0': t0_best_lower_error, 'p': p_best_lower_error, 'a': a_best_lower_error, 'i': i_best_lower_error, 'i_in_degree': i_in_degree_best_lower_error, 'ldc1': ldc1_best_lower_error, 'ldc2': ldc2_best_lower_error, 'ldc': ldc_best_lower_error,
                                        'transit_duration': transit_duration_best_lower_error, 'transit_depth': transit_depth_best_lower_error})
        params_best_upper_error.update({'k': k_best_upper_error, 't0': t0_best_upper_error, 'p': p_best_upper_error, 'a': a_best_upper_error, 'i': i_best_upper_error, 'i_in_degree': i_in_degree_best_upper_error, 'ldc1': ldc1_best_upper_error, 'ldc2': ldc2_best_upper_error, 'ldc': ldc_best_upper_error,
                                        'transit_duration': transit_duration_best_upper_error, 'transit_depth': transit_depth_best_upper_error})


        # Calculate the best fitted model flux, residuals, residual standard deviation, chi-square and reduced chi-square
        lc_fitted = lc.copy()
        if fit_range == 'multi':
            lc_fitted.flux = model_func(params_best_array, tm, time=lc_fitted.time.value) * lc.flux.unit
        elif fit_range == 'single':
            lc_fitted.flux = model_func(params_best_array, p_best, tm, time=lc_fitted.time.value) * lc.flux.unit
        lc_fitted.flux_err = np.zeros(len(lc_fitted.flux_err))

        lc_residual = lc.copy()
        lc_residual.flux = lc.flux - lc_fitted.flux
        lc_residual.flux_err = lc.flux_err - lc_fitted.flux_err

        residual_std = float(np.std(lc_residual.flux.value))
        chi_square = float(np.sum((lc_residual.flux.value / lc.flux_err.value) ** 2))
        reduced_chi_square = float(chi_square / (len(lc.flux.value) - n_dim))


        # Remove the outliers
        _, outliers_mask = lc_residual.remove_outliers(sigma=sigma, return_mask=True)

        if np.sum(outliers_mask) == 0:
            break
        else:
            # update the lightcurve, initial parameters and transit model
            lc = lc[~outliers_mask]
            params_initial = update_dict(params_initial, params_best)
            tm.set_data(lc.time.value)


    params_samples_thinned_unflattened = params_sampler.get_chain(discard=int(n_steps * chain_discard_proportion), thin=chain_thin, flat=False)  # retrieve the thinned and unflattened sample chains from the MCMC sampler
    results = update_dict(results, {
        'n_iterations': n_iter,
        'n_dimensions': n_dim, 'params_name': params_name, 'params_samples': params_samples, 'params_samples_thinned_unflattened': params_samples_thinned_unflattened,
        'params_best': params_best, 'params_best_lower_error': params_best_lower_error, 'params_best_upper_error': params_best_upper_error,
        'lc': lc, 'lc_fitted': lc_fitted, 'lc_residual': lc_residual,
        'residual_std': residual_std, 'chi_square': chi_square, 'reduced_chi_square': reduced_chi_square
    })


    return results


def split_indiviual_lc(lc, p, t0, transit_duration, individual_transit_check_coefficient=1.0):
    """
    Split the light curve into individual transit light curves based on the provided `p`, `t0` and `transit_duration`.

    Parameters
    -----------
    lc : `lightkurve.LightCurve`
        The `LightCurve` object.
    p : float
        The orbital period of the planet in `days`.
    t0 : float
        The epoch time of the planet in `BTJD`.
    transit_duration : float
        The transit duration of the planet in `days`.
    individual_transit_check_coefficient : float, optional
        The coefficient of transit duration span to check if the individual transit light curve contains transit event.
        Default is `1.0`.

    Returns
    --------
    lc_individual_list : list of `lightkurve.LightCurve`
        A list of individual transit `LightCurve` objects.
    transit_info : dict
        A dictionary containing the information of the transits,
        including `n_possible_transits`, `n_valid_transits`, `no_data_transit_indices`, `valid_transit_indices`.
    """
    # Initialize the lc_individual_list and transit_info
    lc_individual_list = []
    transit_info = {'n_possible_transits': None, 'n_valid_transits': None, 'no_data_transit_indices': None, 'valid_transit_indices': None}
    no_data_transit_indices = []
    valid_transit_indices = []


    # Get time range of the light curve
    time_min = np.nanmin(lc.time.value)
    time_max = np.nanmax(lc.time.value)

    # Ensure t0 is the first epoch time in the light curve time range
    while t0 - p >= time_min:
        t0 -= p
    while t0 + p <= time_min + p:
        t0 += p

    # Calculate the first transit window
    transit_half_duration = transit_duration / 2 * individual_transit_check_coefficient
    transit_first_start = t0 - transit_half_duration
    transit_first_end = t0 + transit_half_duration

    # Calculate the range of transit indices
    first_possible_transit_index = int(np.ceil((time_min - transit_first_end) / p)) # first transit index that could potentially be in the light curve
    last_possible_transit_index = int(np.floor((time_max - transit_first_start) / p)) # last transit index that could potentially be in the light curve

    # Check if any transits are possible
    if first_possible_transit_index > last_possible_transit_index:
        raise ValueError("No transit events found in the light curve time range. Check transit parameters and light curve time range.")

    n_possible_transits = last_possible_transit_index - first_possible_transit_index + 1
    possible_transit_indices = list(range(first_possible_transit_index, last_possible_transit_index + 1))


    for transit_index in possible_transit_indices:
        transit_center = t0 + p * transit_index
        transit_start = transit_center - transit_half_duration
        transit_end = transit_center + transit_half_duration

        # Extract the individual transit light curve
        transit_mask = (lc.time.value >= transit_center - p/2) & (lc.time.value <= transit_center + p/2)
        lc_individual = lc[transit_mask]

        # Check if the individual transit light curve has data points
        if len(lc_individual.time) == 0:
            warnings.warn(f"No data points found for transit index {transit_index}. This transit will be skipped.")
            no_data_transit_indices.append(transit_index)
            lc_individual_list.append(None)
        # Check if the individual transit light curve contains transit event
        elif not(np.any((lc_individual.time.value >= transit_start) & (lc_individual.time.value <= transit_end))):
            warnings.warn(f"No transit event found in the light curve for transit index {transit_index}. This transit will be skipped.")
            no_data_transit_indices.append(transit_index)
            lc_individual_list.append(None)
        else:
            valid_transit_indices.append(transit_index)
            lc_individual_list.append(lc_individual)

    n_valid_transits = len(valid_transit_indices)


    transit_info = update_dict(transit_info, {'n_possible_transits': n_possible_transits, 'n_valid_transits': n_valid_transits, 'no_data_transit_indices': no_data_transit_indices, 'valid_transit_indices': valid_transit_indices})


    return lc_individual_list, transit_info


def plot_trace_evolution(results, running_mean_window_length=20):
    """
    Plot the trace and evolution of `MCMC` parameters given the `MCMC` fitting results.

    Parameters
    -----------
    results : dict
        The `MCMC` fitting results dictionary.
    running_mean_window_length : int, optional
        Window length of the thinned-unflattened `MCMC` chain to calculate the running means of the parameters. Default is `20`.

    Returns
    --------
    fig : `matplotlib.figure.Figure`
        The matplotlib `Figure` object with the trace and evolution plots.
    """
    # Retrieve MCMC fitting parameters from the results dictionary
    n_dim = results['n_dimensions']
    params_name = results['params_name']
    params_samples_thinned_unflattened = results['params_samples_thinned_unflattened']
    n_walkers = params_samples_thinned_unflattened.shape[1]

    # Calculate running mean for each parameter
    params_running_means = []
    for d in range(n_dim):
        param_walkers_mean = np.mean(params_samples_thinned_unflattened[:, :, d], axis=1)
        param_running_mean = np.convolve(param_walkers_mean, np.ones(running_mean_window_length) / running_mean_window_length, mode='valid')
        params_running_means.append(param_running_mean)

    # Create figure and grid
    fig = plt.figure(figsize=(20, 2 * n_dim))
    gs = GridSpec(n_dim, 2, wspace=0.05)

    # Plot trace and evolution for each parameter
    for d in range(n_dim):
        # Plot parameter trace on the left side
        ax_trace = fig.add_subplot(gs[d, 0])
        for w in range(n_walkers):
            ax_trace.plot(params_samples_thinned_unflattened[:, w, d], alpha=0.5, linewidth=0.8)
        ax_trace.set_ylabel(params_name[d])
        ax_trace.grid(True, alpha=0.3)
        # only show x-axis label on the last subplot
        if d == n_dim - 1:
            ax_trace.set_xlabel("Step Number")
        else:
            ax_trace.tick_params(labelbottom=False)
        # only show title on the first subplot
        if d == 0:
            ax_trace.set_title("Trace Of Parameters", fontsize='x-large')

        # Plot parameter evolution on the right side
        ax_evolution = fig.add_subplot(gs[d, 1], sharey=ax_trace)
        ax_evolution.plot(params_running_means[d], c='red')
        ax_evolution.tick_params(labelleft=False) # hide y-axis labels
        ax_evolution.grid(True, alpha=0.3)
        # only show x-axis label on the last subplot
        if d == n_dim - 1:
            ax_evolution.set_xlabel("Step Number")
        else:
            ax_evolution.tick_params(labelbottom=False)
        # only show title on the first subplot
        if d == 0:
            ax_evolution.set_title(f"Evolution ({running_mean_window_length}% Step Running Mean) Of Parameters", fontsize='x-large')

    # Set main title
    fig.suptitle(f"Fitted Transit Parameters Trace and Evolution Plot", fontsize='xx-large')
    fig.subplots_adjust(wspace=0.05)

    return fig

def plot_posterior_corner(results, quantiles=[0.16, 0.5, 0.84], **kwargs):
    """
    Plot the `MCMC` parameters posterior distribution `corner` plot given the `MCMC` fitting results.

    Parameters
    -----------
    results : dict
        The `MCMC` fitting results dictionary.
    quantiles : iterable, optional
        A list of fractional quantiles to show on the 1D histograms as vertical dashed lines.
        Will be passed to `corner.corner()` as the `quantiles` argument. Default is `[0.16, 0.5, 0.84]`.
    kwargs : dict
        Dictionary of additional arguments to be passed to `corner.corner()`.

    Returns
    --------
    fig : `matplotlib.figure.Figure`
        The matplotlib `Figure` object with the `corner` plot.
    """
    # Retrieve MCMC fitting parameters from the results dictionary
    params_samples = results['params_samples']
    params_name = results['params_name']

    # Create corner plot
    fig = corner.corner(params_samples, labels=params_name, quantiles=quantiles, **kwargs)

    # Set main title
    fig.suptitle(f"Fitted Transit Parameters Posterior Distribution Corner Plot", fontsize='xx-large', y=1.05)

    return fig
