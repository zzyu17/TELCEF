import yaml
from ruamel.yaml import YAML
import shutil
import re
import numbers
from collections.abc import Mapping, Iterable
from collections import OrderedDict
import os
import sys
import subprocess
import warnings
import time

import astropy.units as u
from astropy.time import Time
from astropy.io import fits
from astropy.table import Table, Column
from astroquery.mast import Observations
from tess_stars2px import tess_stars2px_function_entry as tess_stars2px
import lightkurve as lk
from eleanor.mast import coords_from_name, tic_from_coords, gaia_from_coords
from pytransit import QuadraticModel, RoadRunnerModel
import emcee
import arviz as az
import numpy as np
import corner
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter




### ------ Mission-specified ------ ###
# Define the attribute map and the stage map for different missions
ATTRIBUTE_MAP = {
    'Kepler': 'quarter',
    'K2': 'campaign',
    'TESS': 'sector'
}

STAGE_MAP = {
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


def update_config(config_path, obj_update, precision=12, delete=False):
    """
    Update the specific key-value pairs in a YAML configuration file with new ones,
    while preserving the original formatting, comments and other contents.
    Supports updating specific indices in lists and nested lists.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    obj_update : dict or list
        When `delete` is `False`, a dictionary where keys are dot-separated key paths and values are the new values to update in the configuration file.
        When `delete` is `True`, a list of keys to delete from the configuration file.
        For updating or deleting specific list indices, use the format: 'key.path[index]' or 'key.path[index].nested_key'.
    precision : int, optional
        Precision for float representation. Default is `12` (avoiding edge cases where rounding affects the last digit because of IEEE-754 floating point behavior).
    delete : bool, optional
        If `True`, delete the keys specified in `obj_update`. Default is `False`.

    Returns
    -------
    config : dict
        The updated configuration dictionary.
    """
    # Validate the type of obj_update
    if not delete and not isinstance(obj_update, dict):
        raise ValueError("When 'delete' is False, 'obj_update' must be a dictionary where keys are dot-separated key paths and values are the new values to update.")
    if delete and not isinstance(obj_update, list):
        raise ValueError("When 'delete' is True, 'obj_update' must be a list of keys to delete.")


    yaml = YAML()

    # Set YAML formatting options
    yaml.preserve_quotes = True  # preserve the quote style
    yaml.indent(mapping=2, sequence=4, offset=2)  # preserve the indentation format


    # Customize float construction
    # define custom float constructor
    def yaml_float_constructor(self, node):
        value = self.construct_scalar(node)
        try:
            # try parsing as a normal float first
            return float(value)
        except ValueError:
            # handle scientific notation
            if 'e' in value.lower():
                return float(value)
            # handle special float values
            elif value.lower() in ['.inf', 'inf']:
                return float('inf')
            elif value.lower() in ['-.inf', '-inf']:
                return float('-inf')
            elif value.lower() in ['.nan', 'nan']:
                return float('nan')
            # handle integer-like floats
            elif '.' not in value:
                return int(value)
            else:
                return float(value)
    # add custom float constructor
    yaml.constructor.add_constructor('tag:yaml.org,2002:float', yaml_float_constructor)

    # Customize float representation
    # define custom float representers
    def decimal_float_representer(dumper, value):
        if value.is_integer():
            # represent as integer
            return dumper.represent_int(int(value))
        elif np.isinf(value):
            formatted = '.inf' if value > 0 else '-.inf'
            return dumper.represent_scalar('tag:yaml.org,2002:float', formatted)
        elif np.isnan(value):
            formatted = '.nan'
            return dumper.represent_scalar('tag:yaml.org,2002:float', formatted)
        else:
            # decimal notation with specified precision, removing trailing zeros
            formatted = f"{value:.{precision}f}".rstrip('0')
            return dumper.represent_scalar('tag:yaml.org,2002:float', formatted)
    # add custom float representers
    yaml.representer.add_representer(float, decimal_float_representer)


    # Load the existing configuration
    with open(config_path, 'r') as f:
        config = yaml.load(f) or {}

    # Apply the updates
    if not delete:
        for key_path, value in obj_update.items():
            keys = _parse_key_path(key_path)
            _deep_update(config, keys, value, delete=delete)
    else:
        for key_path in obj_update:
            keys = _parse_key_path(key_path)
            _deep_update(config, keys, delete=delete)

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
            if key:
                # only add if the key is not empty
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

def _deep_update(d, keys, value=None, delete=False):
    """
    A helper function to recursively update or delete from a nested dict `d` at `keys`,
    supporting list indices in the key path.

    Parameters
    ----------
    d : dict or list
        The dictionary or list to update.
    keys : list
        List of keys and indices to traverse.
    value : any, optional
        The value to set at the final key/index. Must be provided if `delete` is `False`.
    delete : bool, optional
        If `True`, delete the specified key. Default is `False`.

    Raises
    ------
    `KeyError`: If a key doesn't exist in a dictionary.
    `IndexError`: If an index is out of range for a list.
    `TypeError`: If trying to index a non-list or access a non-dict.
    """
    # Validate input parameters
    if not keys:
        return
    if not delete and value is None:
        raise ValueError("Value must be provided when 'delete' is `False`.")

    current_key = keys[0]
    remaining_keys = keys[1:]

    if not remaining_keys:
        # Set the value at or delete the final key
        if isinstance(d, dict):
            if delete:
                if current_key in d:
                    del d[current_key]
            else:
                d[current_key] = value
        elif isinstance(d, list):
            if isinstance(current_key, int):
                if current_key < len(d):
                    if delete:
                        del d[current_key]
                    else:
                        d[current_key] = value
                else:
                    raise IndexError(f"Index {current_key} out of range for list of length {len(d)}.")
            elif isinstance(current_key, str):
                raise TypeError(f"Cannot use string key '{current_key}' with list.")
        else:
            raise TypeError(f"Cannot set value on type {type(d)} at or delete key '{current_key}'.")

    else:
        # Traverse deeper at the current key when it's not the final one
        if isinstance(d, dict):
            if current_key in d:
                _deep_update(d[current_key], remaining_keys, value, delete=delete)
            else:
                raise KeyError(f"Key '{current_key}' not found in dictionary.")

        elif isinstance(d, list):
            if isinstance(current_key, int):
                if current_key < len(d):
                    _deep_update(d[current_key], remaining_keys, value, delete=delete)
                else:
                    raise IndexError(f"Index {current_key} out of range for list of length {len(d)}.")
            elif isinstance(current_key, str):
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

def update_dict_none(dict1, dict2, ordered_keys=None, return_map=False):
    """
    Update `dict1` with `dict2` only on keys where `dict1` has `None` values or is missing.

    Parameters
    ----------
    ordered_keys : list, optional
        A list specifying the order of keys in the updated dictionary.
        If `None`, the order of `dict1` followed by new keys from `dict2` is used.
    return_map : bool, optional
        Whether to return a map of updated keys and their new values. Default is `False`.

    Returns
    -------
    updated_dict : dict
        The updated dictionary.
    updated_map : dict or `None`
        A dictionary of keys that were updated and their new values from `dict2` if `return_map` is `True`, `None` otherwise.
    """
    updated_dict = {}
    updated_keys = []

    # Process keys from dict2 in their original order
    for key, value in dict2.items():
        # Check if this key should be updated in dict1
        if dict1.get(key, None) is None and value is not None:
            updated_dict[key] = value
            if return_map:
                updated_keys.append(key)
        else:
            # Use the value from dict1 if it exists
            if key in dict1:
                updated_dict[key] = dict1[key]

    # Add any remaining keys from dict1 that weren't in dict2
    for key, value in dict1.items():
        if key not in updated_dict:
            updated_dict[key] = value

    # Sort the updated dict to have the same order as dict1 and dict2
    if ordered_keys is None:
        ordered_keys = list(dict1.keys()) + [k for k in dict2.keys() if k not in dict1]
    updated_dict = OrderedDict((key, updated_dict[key]) for key in ordered_keys if key in updated_dict)

    # Create an updated map for the keys that were updated
    if return_map:
        if updated_keys:
            updated_map = {key: dict2[key] for key in updated_keys}
        else:
            updated_map = None
        return updated_dict, updated_map

    else:
        return updated_dict


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
        The type of the `FITS` file, `'tpf'` & `'targetpixelfile'` for target pixel file `FITS` file, and `'lc'` & `'lightcurve'` for light curve `FITS` file.
    config : dict
        A dictionary containing the metadata of the source, should at least include TIC ID and Gaia ID of the source, mission, sector, author and exptime.

    Returns
    ----------
    `True` if all metadata match, `False` otherwise.
    """
    # Validate the type parameter
    if type.lower() not in ['tpf', 'targetpixelfile', 'lc', 'lightcurve']:
        raise ValueError(f"Invalid 'type' parameter: {type}. Must be 'tpf' & 'targetpixelfile' (for target pixel file FITS file), or 'lc' & 'lightcurve' (for light curve FITS file).")


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
        The type of the `FITS` file, `'tpf'` & `'targetpixelfile'` for target pixel file `FITS` file, and `'lc'` & `'lightcurve'` for light curve `FITS` file.
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
        - `type` : str, the type of the `FITS` file, `'tpf'` & `'targetpixelfile'` (for target pixel file `FITS` file),
                `'lc'` & `'lightcurve'` (for light curve `FITS` file), or `'targetdata'` (for `eleanor.TargetData` `FITS` file)
        - `name` : str, the name of the source
        - `mission` : str, the mission name, e.g., 'TESS'
        - `sector` : int, the sector number
        - `author` : str, the author of the data
        - `exptime` : int, the exposure time of the data in seconds
        - `tpf_height` : int, optional, the height of the target pixel file in pixels (required if `type` is `tpf` or `targetpixelfile`)
        - `tpf_width` : int, optional, the width of the target pixel file in pixels (required if `type` is `tpf` or `targetpixelfile`)
        - `aperture_mask_type` : str, optional, the type of the aperture mask used when extracting light curve via `lightkurve.TESSTargetPixelFile.extract_aperture_photometry()` method (required if `author` contains `lightkurve_aperture`)
        - `flux_method` : str, optional, the flux extraction method used when extracting light curve via `lightkurve.TESSTargetPixelFile.extract_aperture_photometry()` method (required if `author` contains `lightkurve_aperture`)
        - `centroid_method` : str, optional, the method of estimating the centroids used when extracting light curve via `lightkurve.TESSTargetPixelFile.extract_aperture_photometry()` method (required if `author` contains `lightkurve_aperture`)
        - `lc_type` : str, optional, the type of the `eleanor` light curve, must be one of `Raw`, `Corrected`, `PCA`, `PSF` (required if `type` is `lc` or `lightcurve` and `author` is `eleanor`)
        - `postcard` : bool, optional, whether the data is obtained from `postcard` (required if `author` is `eleanor`)

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
        raise ValueError(f"Invalid 'type' parameter: {type}. Must be 'tpf' & 'targetpixelfile' (for target pixel file FITS file), 'lc' & 'lightcurve' (for light curve FITS file), or 'targetdata' (for eleanor.TargetData FITS file).")

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
                raise ValueError(f"Invalid 'lc_type' parameter: {lc_type}. Must be one of ['Raw', 'Corrected', 'PCA', 'PSF'].")


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
        raise ValueError(f"'fits_file' must be a file path or `HDUList` object. Got {type(fits_file)} instead.")


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
        The matplotlib Axes object to plot on. If `None`, a new one will be generated.
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
        If `None`, will be calculated from the time array of the light curve.
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

        time_sorted = time[sort_indices]
        flux_sorted = flux[sort_indices]
        flux_err_sorted = flux_err[sort_indices]

        lc_sorted = lc.copy()
        lc_sorted.time = time_sorted
        lc_sorted.flux = flux_sorted
        lc_sorted.flux_err = flux_err_sorted

        return lc_sorted


def estimate_lc_flux_err(lc, mask=None, method='mad_diff', new_column=False, new_column_name=None):
    """
    Estimate flux errors for a light curve lacking `flux_err` values using a specified method.

    Parameters
    ----------
    lc : `lightkurve.LightCurve`
        The `LightCurve` object.
    mask : boolean array with length of `lc.time`
        Boolean array to mask data with before estimating the flux errors.
        `Flux` values where `mask` is `True` will not be used to estimate the flux errors.
        Use this mask to remove data with high deviations or known events, e.g. transits.
    method : str, optional
        The method to estimate the flux errors. Currently supported methods are:
        - `'mad_diff'`: Median Absolute Deviation (MAD) method using the differential estimator.
                     Estimates the flux errors using the differential estimator: `flux_err = 1.4826 * mad(diff(flux)) / sqrt(2) * sqrt(flux_normalized)`.
        - `'std_diff'`: Standard Deviation (STD) method using the differential estimator.
                     Estimates the flux errors using the differential estimator: `flux_err = std(diff(flux)) / sqrt(2) * sqrt(flux_normalized)`.
        - `'mad'`: Median Absolute Deviation (MAD) method.
                        Estimates the flux errors by: `flux_err = 1.4826 * mad(flux) * sqrt(flux_normalized)`.
        - `'std'`: Standard Deviation (STD) method.
                        Estimates the flux errors by: `flux_err = std(flux) * sqrt(flux_normalized)`.
        Default is `'mad_diff'`.
    new_column : bool, optional
        Whether to save the estimated flux errors to a new column in the `LightCurve` object. Default is `False`.
    new_column_name : str or None, optional
        The name of the new column to save the estimated flux errors.
        If `None`, the new column will be named `f'flux_err_{method}'`. Default is `None`.

    Returns
    -------
    lc: `lightkurve.LightCurve`
        The `LightCurve` object with updated `flux_err` values.
    """
    # Check if all the light curve `flux_err` values are NaNs
    if not np.all(np.isnan(lc.flux_err.value)) and not new_column:
        notnan_idx = np.where(~np.isnan(lc.flux_err.value))[0]
        warnings.warn(f"The input light curve already has {len(notnan_idx)} non-NaN flux_err values at indices (truncated to first 10 if more than 10): {notnan_idx[:10] if len(notnan_idx) > 10 else notnan_idx}."
                      f"The flux_err values of these data points will be overwritten by the estimated flux_err values.")

    # Validate method
    if method.lower() not in ['mad_diff', 'std_diff', 'mad', 'std']:
        raise ValueError(f"Unsupported method: {method}. Currently supported methods are: 'mad_diff', 'std_diff', 'mad', 'std'.")

    # Validate new_column_name
    if new_column_name is not None and not new_column_name.endswith('_err'):
        raise ValueError(f"The 'new_column_name' (if provided) must contain '_err' suffix to allow to be read as an affiliate `flux_err` column by Lightkurve. Got '{new_column_name}' instead.")


    lc_estimated = lc.copy()
    lc_normalized = lc.copy()

    if mask is not None:
        lc_masked = lc_estimated[~mask]
        lc_normalized.flux /= np.nanmedian(lc_normalized.flux[~mask])
    else:
        lc_masked = lc_estimated.copy()
        lc_normalized.flux /= np.nanmedian(lc_normalized.flux)

    flux_diff = np.diff(lc_masked.flux.value)


    # Estimate flux errors using the specified method
    if method.lower() == 'mad_diff':
        mad_flux_diff = np.nanmedian(np.abs(flux_diff - np.nanmedian(flux_diff)))
        std_flux_diff = 1.4826 * mad_flux_diff
        estimated_flux_err = [std_flux_diff / np.sqrt(2) * np.sqrt(flux_normalized) * lc_estimated.flux.unit for flux_normalized in lc_normalized.flux.value]

    elif method.lower() == 'std_diff':
        std_flux_diff = np.nanstd(flux_diff)
        estimated_flux_err = [std_flux_diff / np.sqrt(2) * np.sqrt(flux_normalized) * lc_estimated.flux.unit for flux_normalized in lc_normalized.flux.value]

    elif method.lower() == 'mad':
        mad_flux = np.nanmedian(np.abs(lc_masked.flux.value - np.nanmedian(lc_masked.flux.value)))
        std_flux = 1.4826 * mad_flux
        estimated_flux_err = [std_flux * np.sqrt(flux_normalized) * lc_estimated.flux.unit for flux_normalized in lc_normalized.flux.value]

    elif method.lower() == 'std':
        std_flux = np.nanstd(lc_masked.flux.value)
        estimated_flux_err = [std_flux * np.sqrt(flux_normalized) * lc_estimated.flux.unit for flux_normalized in lc_normalized.flux.value]


    # Save the estimated flux errors to a new column, also save the original flux to a new column whose name corresponds to the new flux_err column name
    if new_column:
        if new_column_name is None:
            new_column_name = f'{method.lower()}_flux_err'
            new_flux_column_name = f'{method.lower()}_flux'
        else:
            new_flux_column_name = new_column_name.strip('_err')
        new_flux_column = Column(data=lc_estimated.flux, name=new_flux_column_name)
        new_flux_err_column = Column(data=estimated_flux_err, name=new_column_name)

        if new_flux_column_name in lc_estimated.columns:
            warnings.warn(f"Column `{new_flux_column_name}` already exists in the `LightCurve` object and will be overwritten.")
            lc_estimated[new_flux_column_name] = new_flux_column
        if new_column_name in lc_estimated.columns:
            warnings.warn(f"Column `{new_column_name}` already exists in the `LightCurve` object and will be overwritten.")
            lc_estimated[new_column_name] = new_flux_err_column
        else:
            lc_estimated.add_column(new_flux_column)
            lc_estimated.add_column(new_flux_err_column)

    else:
        lc_estimated.flux_err = estimated_flux_err


    return lc_estimated


def alpha_exptime_default(exptime):
    """
    Return the default plotting alpha coefficient of the light curve and residuals corresponding to the exposure time.
    """
    if exptime <= 80:
        alpha_exptime_default = 0.1
    elif 80 < exptime <= 400:
        alpha_exptime_default = 0.3
    elif exptime > 400:
        alpha_exptime_default = 0.5

    return alpha_exptime_default

def scatter_point_size_exptime_default(exptime):
    """
    Return the default scatter point size coefficient of the light curve and residuals corresponding to the exposure time.
    """
    if exptime <= 80:
        scatter_point_size_exptime_default = 1
    elif 80 < exptime <= 400:
        scatter_point_size_exptime_default = 10
    elif exptime > 400:
        scatter_point_size_exptime_default = 20

    return scatter_point_size_exptime_default




### ------ Transit Fitting ------ ###
# Define the transit parameter names
PARAMS_NAME = ['k', 't0', 'p', 'a', 'i', 'ldc1', 'ldc2']
PARAMS_NAME_FULL = ['k', 't0', 'p', 'a', 'i', 'i_in_degree', 'ldc1', 'ldc2', 'ldc', 'transit_duration', 'transit_depth']

# Define the default initial values for transit parameters
PARAMS_INITIAL_DEFAULT = {'k': 0.1, 't0': 3000.0, 'p': 4.0, 'a': 10.0, 'i': np.pi / 2, 'ldc1': 0.2, 'ldc2': 0.3}


### Transit Model ###
def parse_transit_model(transit_model_name):
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


def model_flux(params, transit_model, time):
    """
    Calculate the transit model light curve `flux` values given transit parameters and a transit model.

    Parameters
    ----------
    params : dict
        A dictionary of all the transit parameters, should contain keys for parameter names, i.e., `'k'`, `'t0'`, `'p'`, `'a'`, `'i'`, `'ldc1'`, `'ldc2'`,
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
    flux_model : array-like
        The modeled `flux` values corresponding to the input `time` array based on the provided `params` and `transit_model`.
    """
    k = params['k']
    t0 = params['t0']
    p = params['p']
    a = params['a']
    i = params['i']
    ldc1 = params['ldc1']
    ldc2 = params['ldc2']
    tm = transit_model

    if isinstance(time, np.ndarray):
        tm.set_data(time=time)
    else:
        tm.set_data(time=time.value)

    flux_model = tm.evaluate(k=k, t0=t0, p=p, a=a, i=i, e=0.0, w=0.0, ldc=[ldc1, ldc2])

    return flux_model


### Prior ###
def _physical_boundaries(lc):
    """
    Define the default physical boundaries for transit parameters based on the provided light curve,
    which are as follows:
    - `k`: (0, +inf)
    - `t0`: (min(lc.time), max(lc.time))
    - `p`: (0, +inf)
    - `a`: (1, +inf)
    - `i`: (-:math:\pi/2, :math:\pi/2)
    - `ldc1`: (0, 1)
    - `ldc2`: (0, 1)
    """
    time_min = np.nanmin(lc.time.value)
    time_max = np.nanmax(lc.time.value)

    physical_boundaries_dict = {
        'k': (0, float('inf')), # normalized planetary radius, i.e., R_p/R_s. Must be positive.
        't0': (time_min, time_max), # epoch time in BTJD. Must be within the time range of the light curve (i.e., the observation period).
        'p': (0, float('inf')), # orbital period in days. Must be positive.
        'a': (1, float('inf')), # normalized semi-major axis, i.e., a/R_s. Must be larger than 1.
        'i': (-np.pi / 2, np.pi / 2), # orbital inclination in radians. Must be within [-pi/2, pi/2].
        'ldc1': (0, 1), # linear limb darkening coefficient. Must be within [0, 1].
        'ldc2': (0, 1) # quadratic limb darkening coefficient. Must be within [0, 1].
    }

    return physical_boundaries_dict


def load_params_initial_from_config(config, fit_stage, lc, params_name=PARAMS_NAME):
    """
    Load the initial fitted transit parameters from the configuration dictionary.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
    fit_stage: str
        Stage of fit, should be `global`, `individual` or `fnb`.
    lc : `lightkurve.LightCurve`
        The `LightCurve` object to be fitted.
    params_name : list of str, optional
        A list of transit parameter names to load. Default is `PARAMS_NAME`.

    Returns
    -------
    params_initial : dict
        A dictionary containing the initial fitted transit parameters.
    """
    # Validate fit_stage
    if fit_stage.lower() not in ['global', 'individual', 'fnb']:
        raise ValueError(f"'fit_stage' must be 'global', 'individual' or 'fnb'. Got '{fit_stage}' instead.")

    # Initialize the initial fitted transit parameters dictionary
    params_initial = {key: None for key in params_name}
    params_initial_fit_stage = {}
    params_initial_general = {}
    params_initial_bls_raw_nans_removed = {}
    params_initial_nasa = {}

    for key in params_name:
        # Define the available initial fitted transit parameter in order of priority
        params_initial_fit_stage[key] = config['transit_fitting'][f'{fit_stage}_intitial_transit_parameters'].get(key, None)
        params_initial_general[key] = config['transit_fitting']['intitial_transit_parameters'].get(key, None)
        params_initial_bls_raw_nans_removed[key] = config['lightkurve']['raw_nans_removed_bls_fitted_parameters'].get(key, None)
        params_initial_nasa[key] = config['planet'].get(key, None)

    params_initial = update_dict_none(params_initial, params_initial_fit_stage, ordered_keys=params_name)
    params_initial = update_dict_none(params_initial, params_initial_general, ordered_keys=params_name)
    params_initial = update_dict_none(params_initial, params_initial_bls_raw_nans_removed, ordered_keys=params_name)
    params_initial = update_dict_none(params_initial, params_initial_nasa, ordered_keys=params_name)


    # Ensure t0 is the first epoch time in the light curve time range
    t0_initial = params_initial['t0']
    p_initial = params_initial['p']

    if t0_initial is not None and p_initial is not None:
        time_min = np.nanmin(lc.time.value)
        while t0_initial - p_initial >= time_min:
            t0_initial -= p_initial
        while t0_initial + p_initial <= time_min + p_initial:
            t0_initial += p_initial
        params_initial['t0'] = t0_initial

    elif p_initial is None:
        warnings.warn("The initial value for 'p' is not provided. Cannot set the initial value for 't0' to the first epoch time in the light curve time range automatically. Please set it manually if needed.")


    return params_initial

def load_priors_from_config(config, fit_stage, lc, params_name=PARAMS_NAME):
    """
    Load the prior distribution strings for transit parameters from the configuration dictionary.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
    fit_stage: str
        Stage of fit, should be `global`, `individual` or `fnb`.
    lc : `lightkurve.LightCurve`
        The `LightCurve` object to be fitted.
    params_name : list of str, optional
        A list of transit parameter names to load. Default is `PARAMS_NAME`.

    Returns
    -------
    priors : dict
        A dictionary containing the prior distribution strings for transit parameters.
    """
    # Validate fit_stage
    if fit_stage.lower() not in ['global', 'individual', 'fnb']:
        raise ValueError(f"'fit_stage' must be 'global', 'individual' or 'fnb'. Got '{fit_stage}' instead.")

    # Initialize the priors dictionary
    priors = {key: None for key in params_name}
    priors_fit_stage = {}
    priors_general = {}

    for key in params_name:
        # Define the available prior in order of priority
        priors_fit_stage[key] = config['transit_fitting'][f'{fit_stage}_priors'].get(key, None)
        priors_general[key] = config['transit_fitting']['priors'].get(key, None)

    priors = update_dict_none(priors, priors_fit_stage, ordered_keys=params_name)
    priors = update_dict_none(priors, priors_general, ordered_keys=params_name)


    # Ensure t0 is the first epoch time in the light curve time range
    t0_prior = priors['t0']
    p_prior = priors['p']

    if t0_prior is not None:
        if p_prior is not None:
            p_prior_type, p_prior_params = _parse_prior_string(p_prior)
            if p_prior_type == 'fixed':
                p_initial = p_prior_params[0]

                t0_prior_type, t0_prior_params = _parse_prior_string(t0_prior)
                time_min = np.nanmin(lc.time.value)
                time_max = np.nanmax(lc.time.value)

                if t0_prior_type == 'fixed':
                    v = t0_prior_params[0]
                    while v - p_initial >= time_min:
                        v -= p_initial
                    while v + p_initial <= time_min + p_initial:
                        v += p_initial
                    priors['t0'] = f'fixed({v})'

                elif t0_prior_type in ['uniform', 'hard_boundaries_uniform']:
                    a, b = t0_prior_params
                    mu = (a + b) / 2.0
                    while mu - p_initial >= time_min:
                        mu -= p_initial
                        a -= p_initial
                        b -= p_initial
                    while mu + p_initial <= time_min + p_initial:
                        mu += p_initial
                        a += p_initial
                        b += p_initial
                    a = max(a, time_min)
                    b = min(b, time_max)
                    priors['t0'] = f"{'u' if t0_prior_type == 'uniform' else 'hu'}({a}, {b})"

                elif t0_prior_type == 'normal':
                    mu, sigma = t0_prior_params
                    while mu - p_initial >= time_min:
                        mu -= p_initial
                    while mu + p_initial <= time_min + p_initial:
                        mu += p_initial
                    if mu - 3 * sigma < time_min:
                        mu = time_min + 3 * sigma
                    if mu + 3 * sigma > time_max:
                        mu = time_max - 3 * sigma
                    priors['t0'] = f'n({mu}, {sigma})'

            else:
                warnings.warn("The prior for 'p' is not a 'fixed' prior. Cannot set 't0' prior to the one where 't0' is the first epoch time in the light curve time range automatically. Please set it manually if needed.")
        else:
            warnings.warn("The prior for 'p' is not provided. Cannot set 't0' prior to the one where 't0' is the first epoch time in the light curve time range automatically. Please set it manually if needed.")


    return priors

def load_params_fold_from_config(config, fit_global=False, params_global_best_all=None):
    """
    Load the folding parameters (i.e., folding period and epoch time) from the configuration dictionary.
    """
    # Validate params_global_best_all
    if fit_global and params_global_best_all is None:
        raise ValueError("When 'fit_global' is True, 'params_global_best_all' must be provided.")

    params_fold_name = ['p', 't0']

    # Initialize the folding parameters dictionary
    params_fold = {key: None for key in params_fold_name}
    params_global_best_fold = {}
    params_initial_bls_raw_nans_removed = {}
    params_initial_nasa = {}

    for key in params_fold_name:
        params_fold[key] = config['transit_fitting'].get(f'{key}_fold', None)
        params_global_best_fold[key] = params_global_best_all.get(key, None) if fit_global else None
        params_initial_bls_raw_nans_removed[key] = config['lightkurve']['raw_nans_removed_bls_fitted_parameters'].get(key, None)
        params_initial_nasa[key] = config['planet'].get(key, None)

    params_fold = update_dict_none(params_fold, params_global_best_fold, ordered_keys=params_fold_name)
    params_fold = update_dict_none(params_fold, params_initial_bls_raw_nans_removed, ordered_keys=params_fold_name)
    params_fold = update_dict_none(params_fold, params_initial_nasa, ordered_keys=params_fold_name)

    return params_fold

def print_params_initial_priors(params_initial, priors, fit_stage, lc):
    """
    Print the initial fitted transit parameters and prior distribution strings.
    """
    params_name_initial = list(params_initial.keys())
    params_name_priors = list(priors.keys())

    print(f"{fit_stage.capitalize()} initial fitted transit parameters:")
    max_key_len_params_initial = max(len(key) for key in params_name_initial)
    for key in params_name_initial:
        param_intial = params_initial.get(key, None)
        param_intitial_str = "None" if param_intial is None else f"{param_intial}"
        print(f"    {key:<{max_key_len_params_initial}} : {param_intitial_str:<35} {_physical_boundaries(lc)[key]}")

    print(f"{fit_stage.capitalize()} priors:")
    max_key_len_priors = max(len(key) for key in params_name_priors)
    for key in params_name_priors:
        prior = priors.get(key, None)
        prior_str = "None" if prior is None else f"{prior}"
        print(f"    {key:<{max_key_len_priors}} : {prior_str:<35} {_physical_boundaries(lc)[key]}")
    print("\n")


def parse_params_initial(lc, params_initial=None, priors=None, max_attempts=100):
    """
    Parse a dictionary of initial fitted transit parameters. If not provided, will be derived from the provided `priors`;
    if `priors` is also not provided, the default initial parameters will be used.

    Parameters
    ----------
    lc : `lightkurve.LightCurve`
        The `LightCurve` object to be fitted.
    params_initial : dict, optional
        A dictionary of all the initial transit parameters, should contain keys for parameter names, i.e., `'k'`, `'t0'`, `'p'`, `'a'`, `'i'`, `'ldc1'`, `'ldc2'`.
        If `None`, the default initial parameters will be used.
    priors : dict, optional
        A dictionary mapping parameter names to prior distribution strings:
        - Keys: parameter names, i.e., `'k'`, `'t0'`, `'p'`, `'a'`, `'i'`, `'ldc1'`, `'ldc2'`
        - Values: prior distribution strings, i.e., `'fixed(v)'`, `'u(a, b)'` & `'U(a, b)'`, `'hu(a, b)'` & `'HU(a, b)'` or `'n(mu, sigma)'` & `'N(mu, sigma)'` (see `_parse_prior_string` for details)
        If `None`, the default physical boundaries will be used as uniform priors.
    max_attempts : int, optional
        Maximum number of attempts to generate a valid initial value within physical boundaries for `uniform` and `normal` priors. Default is `100`.

    Returns
    -------
    params_initial : dict
        The parsed `params_initial` dictionary.
    """
    # Define params_initial derived from priors
    params_initial_priors = params_initial_from_priors(lc, priors, max_attempts)

    # If params_initial is not provided, use params_initial derived from priors if available; otherwise, use default params_initial
    if params_initial is None:
        if priors is not None:
            params_initial = params_initial_priors
        else:
            params_initial = PARAMS_INITIAL_DEFAULT

    # Remove unexpected keys
    unexpected_keys = set(params_initial.keys()) - set(PARAMS_INITIAL_DEFAULT.keys())
    if unexpected_keys:
        warnings.warn(f"'params_initial' contains unexpected keys: {sorted(list(unexpected_keys))}. These keys will be ignored.")
        for key in unexpected_keys:
            params_initial.pop(key)

    # Update params_initial for missing or None values
    params_initial, updated_map = update_dict_none(params_initial, params_initial_priors if priors is not None else PARAMS_INITIAL_DEFAULT, ordered_keys=PARAMS_NAME, return_map=True)
    if updated_map:
        warnings.warn(f"'params_initial' lacks or contains `None` values for keys: {sorted(list(updated_map.keys()))}. The default initial parameters will be used for these keys: {updated_map}.")

    return params_initial

def _parse_prior_string(prior_string):
    """
    Parse a prior distribution string and return the distribution type and parameters.
    If no custom prior distribution is provided, the default physical boundaries will be used as uniform prior.

    Parameters
    ----------
    prior_string : str, `'fixed(v)'`, `'u(a, b)'` & `'U(a, b)'`, `'hu(a, b)'` & `'HU(a, b)'` or `'n(mu, sigma)'` & `'N(mu, sigma)'`
        Prior distribution string in the format:
        - `'fixed(v)'` for a fixed value `v`
        - `'u(a, b)'` & `'U(a, b)'` for uniform distribution between `a` and `b`
        - `'hu(a, b)'` & `'HU(a, b)'` for hard boundaries uniform distribution between `a` and `b`
        - `'n(mu, sigma)'` & `'N(mu, sigma)'` for normal distribution with mean `mu` and standard deviation `sigma`

    Returns
    -------
    prior_type : str
        Prior distribution type, `'fixed'`, `'uniform'` or `'normal'`.
    prior_params : tuple
        Prior distribution parameters, `(v,)` for fixed value, `(a, b)` for uniform distribution and `(mu, sigma)` for normal distribution.
    """
    # Remove whitespace and convert to lowercase
    prior_string_clean = ''.join(str(prior_string).split()).lower()

    # Match fixed value: fixed(v)
    fixed_match = re.match(r'^fixed\(([^)]+)\)$', prior_string_clean)
    if fixed_match:
        v = float(fixed_match.group(1))
        prior_type = 'fixed'
        prior_params = (v,)
        return prior_type, prior_params

    # Match uniform distribution: u(a,b) or U(a,b)
    uniform_match = re.match(r'^u\(([^)]+),([^)]+)\)$', prior_string_clean)
    if uniform_match:
        a = float(uniform_match.group(1))
        b = float(uniform_match.group(2))
        if a >= b:
            raise ValueError("Uniform distribution lower bound (a) must be smaller than upper bound (b).")
        prior_type = 'uniform'
        prior_params = (a, b)
        return prior_type, prior_params

    # Match hard boundaries uniform distribution: hu(a,b) or HU(a,b)
    hard_boundaries_uniform_match = re.match(r'^hu\(([^)]+),([^)]+)\)$', prior_string_clean)
    if hard_boundaries_uniform_match:
        a = float(hard_boundaries_uniform_match.group(1))
        b = float(hard_boundaries_uniform_match.group(2))
        if a >= b:
            raise ValueError("Hard boundaries uniform distribution lower bound (a) must be smaller than upper bound (b).")
        prior_type = 'hard_boundaries_uniform'
        prior_params = (a, b)
        return prior_type, prior_params

    # Match normal distribution: n(mu,sigma) or N(mu,sigma)
    normal_match = re.match(r'^n\(([^)]+),([^)]+)\)$', prior_string_clean)
    if normal_match:
        mu = float(normal_match.group(1))
        sigma = float(normal_match.group(2))
        if sigma <= 0:
            raise ValueError("Normal distribution standard deviation (sigma) must be positive.")
        prior_type = 'normal'
        prior_params = (mu, sigma)
        return prior_type, prior_params

    raise ValueError(f"Invalid 'prior_string' parameter: {prior_string}. If provided (i.e., not `None`), must be 'fixed(v)', 'u(a,b)' & 'U(a,b)', 'hu(a, b)' & 'HU(a, b)' or 'n(mu,sigma)' & 'N(mu,sigma)'.")

def parse_priors(lc, priors=None):
    """
    Parse a dictionary of prior distribution strings for transit parameters.

    Parameters
    ----------
    lc : `lightkurve.LightCurve`
        The `LightCurve` object to be fitted.
    priors : dict, optional
        A dictionary mapping parameter names to prior distribution strings:
        - Keys: parameter names, i.e., `'k'`, `'t0'`, `'p'`, `'a'`, `'i'`, `'ldc1'`, `'ldc2'`
        - Values: prior distribution strings, i.e., `'fixed(v)'`, `'u(a, b)'` & `'U(a, b)'`, `'hu(a, b)'` & `'HU(a, b)'` or `'n(mu, sigma)'` & `'N(mu, sigma)'` (see `_parse_prior_string` for details)
        If `None`, the default physical boundaries will be used as uniform priors.

    Returns
    -------
    priors : dict
        The parsed `priors` dictionary.
    """
    # Define default physical boundaries as uniform priors
    physical_boundaries_dict = _physical_boundaries(lc)
    priors_default = {}
    for key in physical_boundaries_dict.keys():
        lower_boundary, upper_boundary = physical_boundaries_dict[key]
        priors_default[key] = f"hu({lower_boundary}, {upper_boundary})"

    # If priors is not provided, use the default physical boundaries as uniform priors
    if priors is None:
        priors = priors_default

    # Remove unexpected keys
    unexpected_keys = set(priors.keys()) - set(priors_default.keys())
    if unexpected_keys:
        warnings.warn(f"'priors' contains unexpected keys: {sorted(list(unexpected_keys))}. These keys will be ignored.")
        for key in unexpected_keys:
            priors.pop(key)

    # Update priors for missing or None values
    priors, updated_map = update_dict_none(priors, priors_default, ordered_keys=PARAMS_NAME, return_map=True)
    if updated_map:
        warnings.warn(f"'priors' lacks or contains `None` values for keys: {sorted(list(updated_map.keys()))}.\n"
                      f"The default physical boundaries will be used as uniform priors for these keys:\n"
                      f"{updated_map}.")

    return priors

def update_t0_prior_individual(t0_prior_first, p, transit_index, lc_individual):
    """
    Update the `t0` prior distribution string for each individual transit fitting.
    """
    if t0_prior_first is None:
        return None

    else:
        # Parse the original t0 prior
        prior_type, prior_params = _parse_prior_string(t0_prior_first)

        time_min = np.nanmin(lc_individual.time.value)
        time_max = np.nanmax(lc_individual.time.value)

        if prior_type == 'fixed':
            v = prior_params[0]
            v += p * transit_index
            if v < time_min:
                v = time_min
            if v > time_max:
                v = time_max
            updated_t0_prior = f"fixed({v})"
            return updated_t0_prior

        elif prior_type in ['uniform', 'hard_boundaries_uniform']:
            a, b = prior_params
            a += p * transit_index
            b += p * transit_index
            a = max(a, time_min)
            b = min(b, time_max)
            updated_t0_prior = f"{'u' if prior_type == 'uniform' else 'hu'}({a}, {b})"
            return updated_t0_prior

        elif prior_type == 'normal':
            mu, sigma = prior_params
            mu += p * transit_index
            if mu - 3 * sigma < time_min:
                mu = time_min + 3 * sigma
            if mu + 3 * sigma > time_max:
                mu = time_max - 3 * sigma
            updated_t0_prior = f"n({mu}, {sigma})"
            return updated_t0_prior

    raise ValueError(f"Invalid 't0_prior_first' parameter: {t0_prior_first}. If provided (i.e., not `None`), must be 'fixed(v)', 'u(a,b)' & 'U(a,b)', 'hu(a, b)' & 'HU(a, b)' or 'n(mu,sigma)' & 'N(mu,sigma)'.")

def update_t0_prior_folded(t0_prior, lc_folded):
    """
    Update the `t0` prior distribution string for folded light curve fitting.
    """
    if t0_prior is None:
        return None

    else:
        # Parse the original t0 prior
        prior_type, prior_params = _parse_prior_string(t0_prior)

        time_min = np.nanmin(lc_folded.time.value)
        time_max = np.nanmax(lc_folded.time.value)

        if prior_type == 'fixed':
            v = 0.0
            updated_t0_prior = f"fixed({v})"
            return updated_t0_prior

        elif prior_type in ['uniform', 'hard_boundaries_uniform']:
            a, b = prior_params
            # shift the center to 0.0
            avg = (a + b) / 2.0
            a -= avg
            b -= avg
            a = max(a, time_min)
            b = min(b, time_max)
            updated_t0_prior = f"{'u' if prior_type == 'uniform' else 'hu'}({a}, {b})"
            return updated_t0_prior

        elif prior_type == 'normal':
            mu, sigma = prior_params
            mu = 0.0
            if - 3 * sigma < time_min:
                sigma = -time_min / 3.0
            if 3 * sigma > time_max:
                sigma = time_max / 3.0
            updated_t0_prior = f"n({mu}, {sigma})"
            return updated_t0_prior

    raise ValueError(f"Invalid 't0_prior' parameter: {t0_prior}. If provided (i.e., not `None`), must be 'fixed(v)', 'u(a,b)' & 'U(a,b)', 'hu(a, b)' & 'HU(a, b)' or 'n(mu,sigma)' & 'N(mu,sigma)'.")


def _param_initial_from_prior(physical_boundaries, prior_string, max_attempts=100):
    """
    A helper function to generate an initial value for a specific parameter given a custom prior distribution.
    If no custom prior distribution is provided, the default physical boundaries will be used as uniform prior.

    Parameters
    ----------
    physical_boundaries : tuple, `(lower_boundary, upper_boundary)`
        Physical boundaries for the parameter.
    prior_string : str, `'fixed(v)'`, `'u(a, b)'` & `'U(a, b)'`, `'hu(a, b)'` & `'HU(a, b)'` or `'n(mu, sigma)'` & `'N(mu, sigma)'`
        Prior distribution string. (See `_parse_prior_string` for details.)
    max_attempts : int, optional
        Maximum number of attempts to generate a valid initial value within physical boundaries for `uniform` and `normal` priors. Default is `100`.

    Returns
    -------
    param_initial : float or None
        An initial value for the specific parameter if a valid initial value can be generated within physical boundaries; otherwise, `None`.
    """
    prior_type, prior_params = _parse_prior_string(prior_string)
    lower_boundary, upper_boundary = physical_boundaries

    if prior_type == 'fixed':
        v = prior_params[0]
        if v < lower_boundary or v > upper_boundary:
            param_initial = None
        else:
            param_initial = v

    elif prior_type in ['uniform', 'hard_boundaries_uniform']:
        def _param_initial_from_uniform_prior(a, b, lower_boundary, upper_boundary):
            if b < lower_boundary or a > upper_boundary:
                param_initial = None
            elif np.isinf(a) or np.isinf(b):
                # if the hard boundaries uniform prior has infinite bounds, set param_initial to None and use default values in PARAMS_INITIAL_DEFAULT later
                param_initial = None
            else:
                param_initial = np.random.uniform(a, b)
            return param_initial
        a, b = prior_params
        param_initial = _param_initial_from_uniform_prior(a, b, lower_boundary, upper_boundary)
        # Ensure the initial value is within physical boundaries (with max_attempts attempts)
        attempts = 0
        while ((param_initial is not None) and (param_initial < lower_boundary or param_initial > upper_boundary)) and attempts < max_attempts:
            attempts += 1
            param_initial = _param_initial_from_uniform_prior(a, b, lower_boundary, upper_boundary)
        if attempts == max_attempts:
            param_initial = None

    elif prior_type == 'normal':
        def _param_intitial_from_normal_prior(mu, sigma):
            param_initial = np.random.normal(mu, sigma)
            return param_initial
        mu, sigma = prior_params
        param_initial = _param_intitial_from_normal_prior(mu, sigma)
        # Ensure the initial value is within physical boundaries (with max_attempts attempts)
        attempts = 0
        while (param_initial < lower_boundary or param_initial > upper_boundary) and attempts < max_attempts:
            attempts += 1
            param_initial = _param_intitial_from_normal_prior(mu, sigma)
        if attempts == max_attempts:
            param_initial = None

    return param_initial

def params_initial_from_priors(lc, priors=None, max_attempts=100):
    """
    Generate initial fitted transit parameters given custom prior distributions.
    If no custom prior distributions are provided, the default physical boundaries will be used as uniform priors.

    Parameters
    ----------
    lc : `lightkurve.LightCurve`
        The `LightCurve` object to be fitted.
    priors : dict, optional
        A dictionary mapping parameter names to prior distribution strings:
        - Keys: parameter names, i.e., `'k'`, `'t0'`, `'p'`, `'a'`, `'i'`, `'ldc1'`, `'ldc2'`
        - Values: prior distribution strings, i.e., `'fixed(v)'`, `'u(a, b)'` & `'U(a, b)'`, `'hu(a, b)'` & `'HU(a, b)'` or `'n(mu, sigma)'` & `'N(mu, sigma)'` (see `_parse_prior_string` for details)
        If `None`, the default physical boundaries will be used as uniform priors.
    max_attempts : int, optional
        Maximum number of attempts to generate a valid initial value within physical boundaries for `uniform` and `normal` priors. Default is `100`.

    Returns
    -------
    params_initial : dict, optional
        A dictionary of all the initial transit parameters, containing keys: `k`, `t0`, `p`, `a`, `i`, `ldc1`, `ldc2`.
    """
    physical_boundaries_dict = _physical_boundaries(lc)
    priors = parse_priors(lc, priors)

    params_initial = {}

    for key in priors.keys():
        physical_boundaries = physical_boundaries_dict[key]
        prior_string = priors[key]
        param_initial = _param_initial_from_prior(physical_boundaries, prior_string, max_attempts)
        if param_initial is None:
            warnings.warn(f"Could not generate a valid initial value for parameter {key} within physical boundaries {physical_boundaries} after {max_attempts} attempts using the provided prior '{prior_string}',\n"
                          f"or the boundaries of the provided prior contains 'inf' value. The initial value for parameter {key} will be set to `None`.")

        params_initial[key] = param_initial

    return params_initial


def _param_log_prior(param, physical_boundaries, prior_string):
    """
    A helper function to calculate the `log-prior` of a specific parameter given a custom prior distribution.
    If no custom prior distribution is provided, the default physical boundaries will be used as uniform prior.

    Parameters
    ----------
    param : float
        The specific parameter.
    physical_boundaries : tuple, `(lower_boundary, upper_boundary)`
        Physical boundaries for the parameter.
    prior_string : str, `'fixed(v)'`, `'u(a, b)'` & `'U(a, b)'`, `'hu(a, b)'` & `'HU(a, b)'` or `'n(mu, sigma)'` & `'N(mu, sigma)'`
        Prior distribution string. (See `_parse_prior_string` for details.)

    Returns
    -------
    log_prior_param : float
        The `log-prior` of a specific parameter.
    """
    # Check if the parameter is within physical boundaries
    lower_boundary, upper_boundary = physical_boundaries
    if param < lower_boundary or param > upper_boundary:
        log_prior_param = -float('inf')

    # Calculate the log-prior based on the specified prior distribution
    prior_type, prior_params = _parse_prior_string(prior_string)

    if prior_type == 'fixed':
        # Fixed value log-prior: 0 if param == v, else -inf
        v = prior_params[0]
        if param == v:
            log_prior_param = 0.0
        else:
            log_prior_param = -float('inf')

    elif prior_type == 'uniform':
        # Uniform distribution log-prior: log(1 / (b - a)) if a <= param <= b, else -inf
        a, b = prior_params
        if a <= param <= b:
            log_prior_param = -np.log(b - a)
        else:
            log_prior_param = -float('inf')

    elif prior_type == 'hard_boundaries_uniform':
        # Hard boundaries uniform distribution log-prior: 0 if a <= param <= b, else -inf
        a, b = prior_params
        if a <= param <= b:
            log_prior_param = 0.0
        else:
            log_prior_param = -float('inf')

    elif prior_type == 'normal':
        # Normal distribution log-prior: -0.5 * log(2 * pi * sigma^2) - 0.5 * ((param - mu) / sigma)^2
        mu, sigma = prior_params
        log_prior_param = -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * ((param - mu) / sigma) ** 2

    return log_prior_param

def model_log_prior(params, lc, priors=None):
    """
    Calculate the `log-prior` of the model parameters given custom prior distributions.
    If no custom prior distributions are provided, the default physical boundaries will be used as uniform priors.

    Parameters
    ----------
    params : dict
        A dictionary of all the transit parameters, should contain keys for parameter names, i.e., `'k'`, `'t0'`, `'p'`, `'a'`, `'i'`, `'ldc1'`, `'ldc2'`.
    lc : `lightkurve.LightCurve`
        The `LightCurve` object to be fitted.
    priors : dict, optional
        A dictionary mapping parameter names to prior distribution strings:
        - Keys: parameter names, i.e., `'k'`, `'t0'`, `'p'`, `'a'`, `'i'`, `'ldc1'`, `'ldc2'`
        - Values: prior distribution strings, i.e., `'fixed(v)'`, `'u(a, b)'` & `'U(a, b)'`, `'hu(a, b)'` & `'HU(a, b)'` or `'n(mu, sigma)'` & `'N(mu, sigma)'` (see `_parse_prior_string` for details)
        If `None`, the default physical boundaries will be used as uniform priors.

    Returns
    -------
    log_prior_model : float
        The `log-prior` of the model parameters.
    """
    physical_boundaries_dict = _physical_boundaries(lc)
    priors = parse_priors(lc, priors)

    log_prior_model = 0.0

    # Calculate log-prior for each parameter
    for k, param in zip(params.keys(), params):
        param = params[k]
        physical_boundaries = physical_boundaries_dict[k]
        prior_string = priors[k]
        param_log_prior = _param_log_prior(param, physical_boundaries, prior_string)

        log_prior_model += param_log_prior

    return log_prior_model


### Likelihood ###
def model_log_likelihood(params, transit_model, lc):
    """
    Calculate the `log-likelihood` of the model given the parameters and the light curve data.

    Parameters
    ----------
    params : dict
        A dictionary of all the transit parameters, should contain keys for parameter names, i.e., `'k'`, `'t0'`, `'p'`, `'a'`, `'i'`, `'ldc1'`, `'ldc2'`.
    transit_model : `PyTransit.TransitModel`
        A `PyTransit` transit model instance (e.g., `QuadraticModel`).
    lc : `lightkurve.LightCurve`
        The `LightCurve` object to be fitted.

    Returns
    -------
    log_likelihood_model : float
        The `log-likelihood` of the model.
    """
    flux_model = model_flux(params, transit_model, lc.time)

    log_likelihood_model = -0.5 * np.sum((lc.flux.value - flux_model) ** 2 / lc.flux_err.value ** 2) - 0.5 * np.sum(np.log(2 * np.pi * lc.flux_err.value ** 2))

    return log_likelihood_model


### Posterior ###
def model_log_posterior(params_free, transit_model, lc, params_fixed=None, priors=None):
    """
    Calculate the `log-posterior` of the model given the parameters and the light curve data.

    Parameters
    ----------
    params_free : dict
        A dictionary of the free transit parameters.
    transit_model : `PyTransit.TransitModel`
        A `PyTransit` transit model instance (e.g., `QuadraticModel`).
    lc : `lightkurve.LightCurve`
        The `LightCurve` object to be fitted.
    params_fixed : dict, optional
        A dictionary of the fixed transit parameters (if any).
    priors : dict, optional
        A dictionary mapping parameter names to prior distribution strings:
        - Keys: parameter names, i.e., `'k'`, `'t0'`, `'p'`, `'a'`, `'i'`, `'ldc1'`, `'ldc2'`
        - Values: prior distribution strings, i.e., `'fixed(v)'`, `'u(a, b)'` & `'U(a, b)'`, `'hu(a, b)'` & `'HU(a, b)'` or `'n(mu, sigma)'` & `'N(mu, sigma)'` (see `_parse_prior_string` for details)
        If `None`, the default physical boundaries will be used as uniform priors.

    Returns
    -------
    log_posterior_model : float
        The `log-posterior` of the model.
    """
    # Merge free and fixed parameters
    if params_fixed is not None:
        params = update_dict_none(params_free, params_fixed)
    else:
        params = params_free

    log_prior_model = model_log_prior(params, lc, priors)
    log_likelihood_model = model_log_likelihood(params, transit_model, lc)

    if np.isfinite(log_prior_model):
        log_posterior_model = log_prior_model + log_likelihood_model
    else:
        log_posterior_model = -float('inf')

    return log_posterior_model


### Transit Properties Calculation ###
def calculate_transit_duration(params):
    """
    Calculate the transit duration (`t_14`) based on the normalized planetary radius (`k`),
    orbital period (`p`), normalized semi-major axis (`a`), and orbital inclination (`i`).

    Parameters
    ----------
    params: dict
        A dictionary of all the transit parameters, should contain keys for parameter names, i.e., `'k'`, `'t0'`, `'p'`, `'a'`, `'i'`, `'ldc1'`, `'ldc2'`.

    Returns
    ----------
    t_14: float
        The transit duration in the same time units as the orbital period (`p`).
    """
    k = params['k']
    p = params['p']
    a = params['a']
    i = params['i']

    b = a * np.cos(i)
    t_14 = np.arcsin(np.sqrt((1 + k) ** 2 - b ** 2) / (a * np.sin(i))) * p / np.pi

    return t_14


def calculate_transit_depth(params, transit_model_name):
    """
    Calculate the transit depth based on the normalized planetary radius (`k`), normalized semi-major axis (`a`),
    orbital inclination (`i`), limb darkening coefficients (`ldc`) and the transit model name.

    Parameters
    ----------
    params: dict
        A dictionary of all the transit parameters, should contain keys for parameter names, i.e., `'k'`, `'t0'`, `'p'`, `'a'`, `'i'`, `'ldc1'`, `'ldc2'`.
    transit_model_name : str
        Name of the `PyTransit` transit model to use. Currently supported models are: `Quadratic`, `RoadRunner_Quadratic`.

    Returns
    ----------
    transit_depth: float
        The normalized transit depth.
    """
    k = np.atleast_1d(params['k'])
    t0 = np.zeros_like(k) # t0 should be set to 0.0
    p = np.ones_like(k) # p can be set randomly
    a = np.atleast_1d(params['a'])
    i = np.atleast_1d(params['i'])
    ldc1 = np.atleast_1d(params['ldc1'])
    ldc2 = np.atleast_1d(params['ldc2'])
    ldc = np.column_stack((ldc1, ldc2))
    if 'roadrunner' in transit_model_name.lower() and ldc.ndim == 2:
        # expand the passband dimension for RoadRunner transit models
        ldc = np.expand_dims(ldc, axis=1)
    tm = parse_transit_model(transit_model_name)

    tm.set_data(time=np.array([0.0])) # calculate the transit depth at t=0.0 (i.e., at the epoch time)

    flux_out = tm.evaluate(k=np.zeros_like(k), t0=t0, p=p, a=a, i=i, ldc=ldc) # calculate the out-of-transit flux at the epoch time, where k=0 means no planet
    flux_in = tm.evaluate(k=k, t0=t0, p=p, a=a, i=i, ldc=ldc) # calculate the in-transit flux at the epoch time
    transit_depth = flux_out - flux_in # calculate the transit depth

    return transit_depth


### Transit Fitting Runner ###
def run_transit_fitting(lc, transit_model_name, fit_type, params_name=PARAMS_NAME, params_initial=None, priors=None,
                        n_walkers=32, n_steps=5000, chain_discard_proportion=0.2, chain_thin=10,
                        max_iter=3, sigma=3.0, transit_index=None):
    """
    Run `MCMC` transit fitting on a light curve using the specified transit model in the specified fit type with outliers removal iterations.

    Parameters
    -----------
    lc : `lightkurve.LightCurve`
        The `LightCurve` object. Must contain `flux` column.
    transit_model_name : str
        Name of the `PyTransit` transit model to use. Currently supported models are: `Quadratic`, `RoadRunner_Quadratic`.
    fit_type : str
        Type of fit, should be `global`, `individual` or `folded`.
    params_name : list of str, optional
        A list of transit parameter names, including both the free and fixed parameters. Default is `PARAMS_NAME`.
    params_initial : dict, optional
        A dictionary of all the initial transit parameters, should contain keys for parameter names, i.e., `'k'`, `'t0'`, `'p'`, `'a'`, `'i'`, `'ldc1'`, `'ldc2'`.
        If `None`, will be derived from the provided `priors`; if `priors` is also `None`, the default initial parameters will be used.
    priors : dict, optional
        A dictionary mapping parameter names to prior distribution strings:
        - Keys: parameter names, i.e., `'k'`, `'t0'`, `'p'`, `'a'`, `'i'`, `'ldc1'`, `'ldc2'`
        - Values: prior distribution strings, i.e., `'fixed(v)'`, `'u(a, b)'` & `'U(a, b)'`, `'hu(a, b)'` & `'HU(a, b)'` or `'n(mu, sigma)'` & `'N(mu, sigma)'` (see `_parse_prior_string` for details)
        If `None`, the default physical boundaries will be used as uniform priors.
    n_walkers : int, optional
        Number of `MCMC` walkers. Default is `32`.
    n_steps : int, optional
        Number of `MCMC` steps. Default is `5000`.
    chain_discard_proportion : float, optional
        Proportion of `MCMC` chain to discard as burn-in. Default is `0.2`.
    chain_thin : int, optional
        Thinning factor of the MCMC chain. Default is `10`.
    max_iter : int, optional
        Maximum number of outliers removal iterations. Default is `3`.
    sigma : float, optional
        `Sigma` threshold for outliers removal. Default is `3.0`.
    transit_index : int, optional
        Index of the transit to be fitted for individual transit fitting.

    Returns
    --------
    results : dict
        A dictionary containing the `MCMC` fitting results, including transit parameters samples, best fitted transit parameters and their uncertainties, best fitted model and residuals, goodness-of-fit metrics and MCMC diagnostics.
    """
    # Parse transit_model_name
    tm = parse_transit_model(transit_model_name)

    # Validate and parse fit_type
    if fit_type.lower() not in ['global', 'individual', 'folded']:
        raise ValueError(f"'fit_type' must be 'global', 'individual' or 'folded'. Got '{fit_type}' instead.")
    if fit_type.lower() != 'individual' and transit_index is not None:
        warnings.warn(f"'transit_index' is only used for individual transit fitting, so it will be ignored for {fit_type.lower()} transit fitting.")
        transit_index = None
    if fit_type.lower() == 'individual' and transit_index is None:
        raise ValueError(f"'transit_index' must be provided for individual transit fitting.")

    # Parse initial fitted transit parameters
    params_initial = parse_params_initial(lc, params_initial, priors)
    params_initial_first_iter = params_initial.copy() # save initial parameters of the first iteration to the results dictionary

    # Validate priors (period should be fixed for individual or folded transit fitting)
    if fit_type.lower() in ['individual', 'folded']:
        if priors is None or 'p' not in priors or priors['p'] is None:
            raise ValueError(f"For {fit_type.lower()} transit fitting, period ('p') must be provided and fixed (i.e., the prior of period ('p') must be set to 'fixed(v)').")
        p_prior_type, _ = _parse_prior_string(priors['p'])
        if p_prior_type != 'fixed':
            raise ValueError(f"For {fit_type.lower()} transit fitting, period ('p') must be fixed (i.e., the prior of period ('p') must be set to 'fixed(v)').")

    # Parse priors
    priors = parse_priors(lc, priors)
    priors_first_iter = priors.copy() # save priors of the first iteration to the results dictionary

    # Identify free and fixed parameters based on prior types
    params_free = {}
    params_fixed = {}
    params_name_free = []
    params_name_fixed = []

    for param_name in params_name:
        prior_type, _ = _parse_prior_string(priors[param_name])
        if prior_type == 'fixed':
            params_fixed[param_name] = params_initial[param_name]
            params_name_fixed.append(param_name)
        else:
            params_free[param_name] = params_initial[param_name]
            params_name_free.append(param_name)

    ## below here, "params" specifically refers to the free parameters only
    params_initial_all = params_initial
    n_params_free = len(params_name_free)


    # Initialize best fitted transit parameters and parameter errors dictionaries
    params_name_full = params_name.copy()
    if 'i' in params_name_full:
        params_name_full.insert(params_name_full.index('i') + 1, 'i_in_degree') # add 'i_degree' after 'i' in params_name_full
    if 'ldc2' in params_name_full:
        params_name_full.insert(params_name_full.index('ldc2') + 1, 'ldc') # add 'ldc' after 'ldc2' in params_name_full
    params_name_full.insert(len(params_name_full), 'transit_duration')
    params_name_full.insert(len(params_name_full), 'transit_depth')

    params_best_full = {key: None for key in params_name_full}
    params_best_lower_error_full = {key: None for key in params_name_full}
    params_best_upper_error_full = {key: None for key in params_name_full}

    # Initialize results dictionary
    results = {
        'n_iterations': 0,
        'params_name_full': params_name_full,
        'n_params_free': n_params_free, 'params_name_free': params_name_free, 'params_name_fixed': params_name_fixed,
        'params_intitial': params_initial_first_iter, 'priors': priors_first_iter,
        'params_samples': None, 'params_samples_unflattened': None,
        'params_best_full': params_best_full, 'params_best_lower_error_full': params_best_lower_error_full, 'params_best_upper_error_full': params_best_upper_error_full,
        'lc': None, 'lc_fitted': None, 'lc_residual': None,
        'residual_std': None, 'chi_square': None, 'reduced_chi_square': None,
        'r_hat': None, 'ess_bulk':None, 'ess_tail': None
    }


    # Initialize the transit model
    tm.set_data(lc.time.value)


    # Run MCMC fitting with outliers removal iterations
    for n_iter in range(1, max_iter + 1):
        # Convert initial parameters dictionary to array for MCMC sampler
        params_initial_array = np.array([params_initial_all[key] for key in params_name_free])

        # Initialize MCMC walkers
        params_position_initial = params_initial_array + 1e-4 * np.random.randn(n_walkers, n_params_free)

        # Set up sampler arguments
        sampler_args = (tm, lc, params_fixed, priors)

        # Set up progress bar description
        progress_desc = f"{fit_type.capitalize()} Fitting"
        if transit_index is not None:
            progress_desc += f" for Transit {transit_index:02}"
        progress_desc += f" Iteration {n_iter}: "

        # Initialize and run the MCMC sampler
        params_sampler = emcee.EnsembleSampler(n_walkers, n_params_free, model_log_posterior, args=sampler_args, parameter_names=params_name_free)
        params_sampler.run_mcmc(params_position_initial, n_steps, progress=True, progress_kwargs={'desc': progress_desc})

        # Retrieve the best fitted transit parameters and their uncertainties from the MCMC sampler
        params_samples_array = params_sampler.get_chain(discard=int(n_steps * chain_discard_proportion), thin=int(chain_thin), flat=True)
        params_samples_unflattened_array = params_sampler.get_chain(discard=int(n_steps * chain_discard_proportion), thin=chain_thin, flat=False) # retrieve the unflattened sample chains from the MCMC sampler
        # convert the values to python floats for YAML configuration file output
        params_best_array = to_python_floats(np.median(params_samples_array, axis=0))
        params_best_lower_error_array = to_python_floats(params_best_array - np.percentile(params_samples_array, 16, axis=0))
        params_best_upper_error_array = to_python_floats(np.percentile(params_samples_array, 84, axis=0) - params_best_array)

        params_samples = {key: params_samples_array[:, i] for i, key in enumerate(params_name_free)}
        params_samples_unflattened = {key: params_samples_unflattened_array[:, :, i] for i, key in enumerate(params_name_free)}
        params_best = {key: params_best_array[i] for i, key in enumerate(params_name_free)}
        params_best_lower_error = {key: params_best_lower_error_array[i] for i, key in enumerate(params_name_free)}
        params_best_upper_error = {key: params_best_upper_error_array[i] for i, key in enumerate(params_name_free)}

        params_samples_all = update_dict_none({key: params_samples.get(key, None) for key in params_name}, {key: np.full((params_samples_array.shape[0],), params_fixed[key]) for key in params_name_fixed}, ordered_keys=params_name)
        params_samples_unflattened_all = update_dict_none({key: params_samples_unflattened.get(key, None) for key in params_name}, {key: np.full((params_samples_unflattened_array.shape[0], params_samples_unflattened_array.shape[1]), params_fixed[key]) for key in params_name_fixed}, ordered_keys=params_name)
        params_best_all = update_dict_none({key: params_best.get(key, None) for key in params_name}, params_fixed, ordered_keys=params_name)
        params_best_lower_error_all = update_dict_none({key: params_best_lower_error.get(key, None) for key in params_name}, {key: 0.0 for key in params_name_fixed}, ordered_keys=params_name)
        params_best_upper_error_all = update_dict_none({key: params_best_upper_error.get(key, None) for key in params_name}, {key: 0.0 for key in params_name_fixed}, ordered_keys=params_name)

        i_in_degree_best, i_in_degree_best_lower_error, i_in_degree_best_upper_error = to_python_floats([np.rad2deg(params_best_all['i']), np.degrees(params_best_lower_error_all['i']), np.degrees(params_best_upper_error_all['i'])])

        ldc_samples = np.column_stack((params_samples_all['ldc1'], params_samples_all['ldc2']))
        ldc_best, ldc_best_lower_error, ldc_best_upper_error = to_python_floats([[params_best_all['ldc1'], params_best_all['ldc2']], [params_best_lower_error_all['ldc1'], params_best_lower_error_all['ldc2']], [params_best_upper_error_all['ldc1'], params_best_upper_error_all['ldc2']]])

        transit_duration_samples = calculate_transit_duration(params_samples_all)
        transit_duration_best = float(np.nanmedian(transit_duration_samples, axis=0))
        transit_duration_best_lower_error = float(transit_duration_best - np.nanpercentile(transit_duration_samples, 16, axis=0))
        transit_duration_best_upper_error = float(np.nanpercentile(transit_duration_samples, 84, axis=0) - transit_duration_best)

        transit_depth_samples = calculate_transit_depth(params_samples_all, transit_model_name)
        transit_depth_best = float(np.nanmedian(transit_depth_samples, axis=0))
        transit_depth_best_lower_error = float(transit_depth_best - np.nanpercentile(transit_depth_samples, 16, axis=0))
        transit_depth_best_upper_error = float(np.nanpercentile(transit_depth_samples, 84, axis=0) - transit_depth_best)

        # Update the best fitted transit parameters and their uncertainties dictionaries
        params_best_full = update_dict(params_best_full, {
            'k': params_best_all['k'], 't0': params_best_all['t0'], 'p': params_best_all['p'], 'a': params_best_all['a'], 'i': params_best_all['i'], 'i_in_degree': i_in_degree_best,
            'ldc1': params_best_all['ldc1'], 'ldc2': params_best_all['ldc2'], 'ldc': ldc_best, 'transit_duration': transit_duration_best, 'transit_depth': transit_depth_best
        })
        params_best_lower_error_full = update_dict(params_best_lower_error_full, {
            'k': params_best_lower_error_all['k'], 't0': params_best_lower_error_all['t0'], 'p': params_best_lower_error_all['p'], 'a': params_best_lower_error_all['a'], 'i': params_best_lower_error_all['i'], 'i_in_degree': i_in_degree_best_lower_error,
            'ldc1': params_best_lower_error_all['ldc1'], 'ldc2': params_best_lower_error_all['ldc2'], 'ldc': ldc_best_lower_error, 'transit_duration': transit_duration_best_lower_error, 'transit_depth': transit_depth_best_lower_error
        })
        params_best_upper_error_full = update_dict(params_best_upper_error_full, {
            'k': params_best_upper_error_all['k'], 't0': params_best_upper_error_all['t0'], 'p': params_best_upper_error_all['p'], 'a': params_best_upper_error_all['a'], 'i': params_best_upper_error_all['i'], 'i_in_degree': i_in_degree_best_upper_error,
            'ldc1': params_best_upper_error_all['ldc1'], 'ldc2': params_best_upper_error_all['ldc2'], 'ldc': ldc_best_upper_error, 'transit_duration': transit_duration_best_upper_error, 'transit_depth': transit_depth_best_upper_error
        })


        # Calculate the best fitted model flux, residuals, residual standard deviation, chi-square and reduced chi-square
        lc_fitted = lc.copy()
        lc_fitted.flux = model_flux(params_best_all, tm, time=lc_fitted.time.value) * lc.flux.unit
        lc_fitted.flux_err = np.zeros(len(lc_fitted.flux_err))

        lc_residual = lc.copy()
        lc_residual.flux = lc.flux - lc_fitted.flux
        lc_residual.flux_err = lc.flux_err - lc_fitted.flux_err

        residual_std = float(np.std(lc_residual.flux.value))
        chi_square = float(np.sum((lc_residual.flux.value / lc.flux_err.value) ** 2))
        reduced_chi_square = float(chi_square / (len(lc.flux.value) - n_params_free))


        # Calculate MCMC diagnostics
        diagnostics = arviz_diagnostics(params_samples_unflattened_all)
        r_hat = diagnostics['r_hat']
        ess_bulk = diagnostics['ess_bulk']
        ess_tail = diagnostics['ess_tail']


        results = update_dict(results, {
            'n_iterations': n_iter,
            'params_samples': params_samples_all, 'params_samples_unflattened': params_samples_unflattened_all,
            'params_best_full': params_best_full, 'params_best_lower_error_full': params_best_lower_error_full, 'params_best_upper_error_full': params_best_upper_error_full,
            'lc': lc, 'lc_fitted': lc_fitted, 'lc_residual': lc_residual,
            'residual_std': residual_std, 'chi_square': chi_square, 'reduced_chi_square': reduced_chi_square,
            'r_hat': r_hat, 'ess_bulk': ess_bulk, 'ess_tail': ess_tail
        })


        # Remove the outliers
        _, outliers_mask = lc_residual.remove_outliers(sigma=sigma, return_mask=True)

        if np.sum(outliers_mask) == 0:
            break
        else:
            # update the lightcurve, initial parameters and transit model
            lc = lc[~outliers_mask]
            params_initial_all = update_dict(params_initial_all, params_best_all)
            tm.set_data(lc.time.value)


    print(f"{fit_type.capitalize()} fitting completed in {n_iter} iteration(s).\n")

    return results


def supersample_lc_fitted(params, transit_model_name, lc, supersample_factor=10, return_residual=True):
    """
    Generate a supersampled best fitted model lightcurve and residuals based on the provided transit parameters and transit model name.

    Parameters
    -----------
    params : dict
        A dictionary of all the transit parameters, should contain keys for parameter names, i.e., `'k'`, `'t0'`, `'p'`, `'a'`, `'i'`, `'ldc1'`, `'ldc2'`.
    transit_model_name : str
        Name of the `PyTransit` transit model to use. Currently supported models are: `Quadratic`, `RoadRunner_Quadratic`.
    lc : `lightkurve.LightCurve`
        The `LightCurve` object. Must contain `flux` column.
    supersample_factor : int, optional
        The supersampling factor, i.e., how many times to supersample the light curve. Default is `10`.
    return_residual : bool, optional
        Whether to return the residual light curve. Default is `True`.

    Returns
    --------
    lc_fitted : `lightkurve.LightCurve`
        The supersampled best fitted model lightcurve.
    lc_residual : `lightkurve.LightCurve`, optional
        The supersampled residual lightcurve. Only returned if `return_residual` is `True`.
    """
    # Parse transit_model_name
    tm = parse_transit_model(transit_model_name)

    # Generate supersampled time array
    time_segments = []
    for i in range(len(lc.time.value) - 1):
        start_time = lc.time.value[i]
        end_time = lc.time.value[i + 1]
        segment = np.linspace(start_time, end_time, supersample_factor, endpoint=False)
        time_segments.append(segment)
    time_segments.append(np.array([lc.time.value[-1]]))
    time = np.concatenate(time_segments)
    lc_time = Time(time, format=lc.time.format, scale=lc.time.scale)


    lc_fitted = lk.LightCurve(time=lc_time,
                              flux=model_flux(params, tm, time=lc_time),
                              flux_err=np.zeros(len(lc_time)))


    if return_residual:
        lc_fitted_ori = lk.LightCurve(time=lc.time,
                                      flux=model_flux(params, tm, time=lc.time.value),
                                      flux_err=np.zeros(len(lc.time)))

        lc_residual = lk.LightCurve(time=lc.time,
                                    flux=lc.flux - lc_fitted_ori.flux,
                                    flux_err=lc.flux_err - lc_fitted_ori.flux_err)

        return lc_fitted, lc_residual

    else:
        return lc_fitted

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


def _drop_fixed_params_from_samples(params_samples):
    """
    A helper function to drop fixed parameters from the `MCMC` fitting samples dictionary.

    Parameters
    -----------
    params_samples: dict
        The `MCMC` fitting samples dictionary.

    Returns
    --------
    params_samples_free: dict
        The `MCMC` fitting samples dictionary with fixed parameters dropped, i.e., only free parameters are kept.
    """
    params_samples_free = {}
    for key, value in params_samples.items():
        if not (value.min() == value.max()):
            params_samples_free[key] = value
    return params_samples_free

def _arviz_idata_from_samples(params_samples):
    """
    A helper function to convert the `MCMC` fitting samples dictionary to an `arviz.InferenceData` object.

    Parameters
    -----------
    params_samples: dict
        The `MCMC` fitting samples dictionary.

    Returns
    --------
    idata : `arviz.InferenceData`
        The `arviz.InferenceData` object containing the `MCMC` fitting samples.
    """
    # Retrieve and validate MCMC samples and parameters from the samples dictionary
    n_params = len(params_samples)
    params_name = list(params_samples.keys())
    if min(params_samples[key].ndim for key in params_samples.keys()) == 1:
        # expand the walker dimension for flattened samples
        for key in params_samples.keys():
            params_samples[key] = np.expand_dims(params_samples[key], axis=params_samples[key].ndim)
    params_samples_array = np.array(list(params_samples.values())).transpose(2, 1, 0) # convert to shape: (n_walkers, n_steps, n_params)

    # Create arviz InferenceData object
    idata = az.from_dict(posterior={params_name[d]: params_samples_array[:, :, d] for d in range(n_params)},)

    return idata

def arviz_diagnostics(params_samples):
    """
    Calculate `arviz` diagnostics for the `MCMC` fitting samples.

    Parameters
    -----------
    params_samples: dict
        The `MCMC` fitting samples dictionary.

    Returns
    --------
    diagnostics : dict
        A dictionary containing the `arviz` diagnostics, including `mcse_mean`, `mcse_sd`, `ess_bulk`, `ess_tail`, and `r_hat`.
    """
    idata = _arviz_idata_from_samples(params_samples)

    diagnostics_df = az.summary(idata, kind='diagnostics')
    diagnostics = to_python_floats(diagnostics_df.to_dict('dict'))  # convert the diagnostics DataFrame to a nested dictionary grouped by diagnostic quantity

    return diagnostics


def plot_trace_evolution(params_samples, running_mean_window_length=20):
    """
    Plot the trace and evolution of `MCMC` parameters given the `MCMC` fitting samples.

    Parameters
    -----------
    params_samples: dict
        The `MCMC` fitting samples dictionary. Recommended to use the unflattened (i.e., multi-walker) samples.
    running_mean_window_length : int, optional
        Window length of the unflattened `MCMC` chain to calculate the running means of the parameters. Default is `20`.

    Returns
    --------
    fig : `matplotlib.figure.Figure`
        The matplotlib `Figure` object with the trace and evolution plots.
    """
    # Retrieve and validate MCMC samples and parameters from the samples dictionary
    params_samples_free = _drop_fixed_params_from_samples(params_samples)
    n_params_free = len(params_samples_free)
    params_name_free = list(params_samples_free.keys())
    if min(params_samples_free[key].ndim for key in params_samples_free.keys()) == 1:
        # expand the walker dimension for flattened samples
        for key in params_samples_free.keys():
            params_samples_free[key] = np.expand_dims(params_samples_free[key], axis=params_samples_free[key].ndim)
    params_samples_free_array = np.array(list(params_samples_free.values())).transpose(1, 2, 0) # convert to shape: (n_steps, n_walkers, n_params)
    n_walkers = params_samples_free_array.shape[1]

    params_name_fixed = [key for key in params_samples if key not in params_name_free]
    n_params_fixed = len(params_name_fixed)
    params_name_fixed_string = ', '.join(params_name_fixed) if len(params_name_fixed) > 0 else 'None'

    # Calculate MCMC diagnostics
    diagnostics = arviz_diagnostics(params_samples_free)
    r_hat = diagnostics['r_hat']
    ess_bulk = diagnostics['ess_bulk']
    ess_tail = diagnostics['ess_tail']

    # Calculate running mean for each parameter
    params_running_mean = []
    for d in range(n_params_free):
        param_walkers_mean = np.mean(params_samples_free_array[:, :, d], axis=1)
        param_running_mean = np.convolve(param_walkers_mean, np.ones(running_mean_window_length) / running_mean_window_length, mode='valid')
        params_running_mean.append(param_running_mean)

    # Create figure and grid
    fig = plt.figure(figsize=(20, 1.5 * (n_params_free + 1)), layout='constrained')
    gs = GridSpec(n_params_free, 2, wspace=0.05, figure=fig)

    # Plot trace and evolution for each parameter
    for d in range(n_params_free):
        param_name_free = params_name_free[d]
        # Plot parameter trace on the left side
        ax_trace = fig.add_subplot(gs[d, 0])
        for w in range(n_walkers):
            ax_trace.plot(params_samples_free_array[:, w, d], alpha=0.5, linewidth=0.8)
        ax_trace.set_ylabel(param_name_free)
        ax_trace.yaxis.set_major_formatter(ScalarFormatter(useOffset=False)) # disable offset scientific notation
        ax_trace.grid(True, alpha=0.3)
        # only show x-axis label on the last subplot
        if d == n_params_free - 1:
            ax_trace.set_xlabel("Step Number")
        else:
            ax_trace.tick_params(labelbottom=False)
        # only show title on the first subplot
        if d == 0:
            ax_trace.set_title("Trace Of Parameters", fontsize='x-large')

        # Plot parameter evolution on the right side
        ax_evolution = fig.add_subplot(gs[d, 1], sharey=ax_trace)
        ax_evolution.plot(params_running_mean[d], label=f"R-hat={r_hat[param_name_free]:.2f}, Bulk-ESS={ess_bulk[param_name_free]:.0f}, Tail-ESS={ess_tail[param_name_free]:.0f}" if n_walkers > 1
                                                        else f"Bulk-ESS={ess_bulk[param_name_free]:.0f}, Tail-ESS={ess_tail[param_name_free]:.0f}", c='red')
        ax_evolution.legend(loc='upper right')
        ax_evolution.tick_params(labelleft=False) # hide y-axis labels
        ax_evolution.grid(True, alpha=0.3)
        # only show x-axis label on the last subplot
        if d == n_params_free - 1:
            ax_evolution.set_xlabel("Step Number")
        else:
            ax_evolution.tick_params(labelbottom=False)
        # only show title on the first subplot
        if d == 0:
            ax_evolution.set_title(f"Evolution ({running_mean_window_length} Steps Running Mean) Of Parameters", fontsize='x-large')

    # Set main title
    fig.suptitle(f"Fitted Transit Parameters Trace and Evolution\n(Fixed Parameters: {params_name_fixed_string})", fontsize='xx-large')

    return fig

def plot_posterior_corner(params_samples, quantiles=[0.16, 0.5, 0.84], **kwargs):
    """
    Plot the `MCMC` parameters posterior distribution `corner` plot given the `MCMC` fitting samples.

    Parameters
    -----------
    params_samples: dict
        The `MCMC` fitting samples dictionary. Must use the flattened (single-chain) samples.
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
    # Retrieve and validate MCMC samples and parameters from the samples dictionary
    params_samples_free = _drop_fixed_params_from_samples(params_samples)
    params_name_free = list(params_samples_free.keys())
    params_samples_free_array = np.array(list(params_samples_free.values())).transpose(1, 0) # convert to shape: (n_steps, n_params)
    if params_samples_free_array.ndim == 3:
        raise ValueError("The provided samples appear to be unflattened (i.e., multi-walker) samples. Please provide flattened (single-chain) samples for corner plot.")

    # Create corner plot
    fig = corner.corner(params_samples_free_array, labels=params_name_free, quantiles=quantiles, **kwargs)

    # Disable offset scientific notation
    for ax in fig.get_axes():
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    # Set main title
    fig.suptitle(f"Fitted Transit Parameters Posterior Distribution Corner Plot", fontsize='xx-large', y=1.05)

    return fig
