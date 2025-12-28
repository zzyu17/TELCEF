import os

from utils import attribute_map, stage_map, load_config, update_config, get_source_metadata, epoch_time_to_btjd
from A_aa_Sector_Locator import *




# Load the configurations
config = load_config(config_path)


##### Define the mission parameters #####
mission = config['mission']['mission']
tess_time = config['mission']['tess_time'] # BJD
attribute = attribute_map.get(mission)
stage = stage_map.get(mission)

##### Define the source parameters #####
name = config['source']['name']
sectors = config['source']['available_sectors']
tpf_exptimes = config['source']['available_tpf_exptimes']
lc_exptimes = config['source']['available_lc_exptimes']
sector = config['source']['sector'] ##### set the single sector to be processed #####
camera = config['source']['camera']
ccd = config['source']['ccd']
ccd_coords = config['source']['ccd_coords']
exptime = config['source']['exptime'] ##### set the exposure time of data to be processed #####
exptime_in_day = exptime / 86400 # convert the exposure time from seconds to days
tic = config['source']['tic']
RA = config['source']['RA']
Dec = config['source']['Dec']
coords = (RA, Dec)
gaia = config['source']['gaia']
tess_mag = config['source']['tess_mag']

##### Define the planet parameters #####
k_nasa = config['planet']['k']
t0_bjd_nasa = config['planet']['t0_bjd']
t0_nasa = config['planet']['t0']
p_nasa = config['planet']['p']
a_nasa = config['planet']['a']
i_nasa = config['planet']['i']
transit_duration_nasa = config['planet']['transit_duration']

cdpp_transit_duration = config['planet']['cdpp_transit_duration'] if config['planet']['cdpp_transit_duration'] is not None else float(config['planet']['transit_duration'] * 24) # convert the transit duration from days to hours


# Validate the 'sector' parameter
if sector not in sectors:
    raise ValueError(f"The specified 'sector' parameter (Sector {sector}) does not exist in the available sectors: {sectors}. Please refer to the SCCE information file for the avaiable sectors, and update the 'sector' parameter accordingly in the configuration file.")

# Validate the 'exptime' parameter
if exptime not in tpf_exptimes[sectors.index(sector)] and exptime not in lc_exptimes[sectors.index(sector)]:
    raise ValueError(f"The specified 'exptime' parameter ({exptime}s) does not exist in the specified sector (Sector {sector}). Please refer to the SCCE information file for the avaiable exptimes, and update the 'exptime' parameter accordingly in the configuration file.")


##### Define the directories #####
# Define the globally used Lightkurve and eleanor directories
lightkurve_cache_root = os.path.expanduser(config['directory']['lightkurve_cache_root'])
lightkurve_cache_root_tesscut = lightkurve_cache_root + config['directory']['lightkurve_cache_root_tesscut']
lightkurve_cache_root_tpf = lightkurve_cache_root + config['directory']['lightkurve_cache_root_tpf']
lightkurve_cache_root_lc = lightkurve_cache_root + config['directory']['lightkurve_cache_root_lc']

eleanor_root = os.path.expanduser(config['directory']['eleanor_root'])
eleanor_root_metadata = eleanor_root + config['directory']['eleanor_root_metadata']
eleanor_root_metadata_sector = eleanor_root_metadata + config['directory']['eleanor_root_metadata_sector']
eleanor_root_tesscut = eleanor_root + config['directory']['eleanor_root_tesscut']

# Define the data directory
data_dir = base_dir + config['directory']['data_dir']
os.makedirs(data_dir, exist_ok=True)




if __name__ == "__main__":
    # Convert epoch time from NASA Exoplanet Archive from BJD to BTJD
    t0_nasa = epoch_time_to_btjd(t0_bjd_nasa, p_nasa)
    config = update_config(config_path, {'planet.t0': t0_nasa})

    # Retrieve the metadata of the source and update the configurations
    metadata_dict = get_source_metadata(name, sector)
    config = update_config(config_path, metadata_dict)

    print(f"Successfully retrieve the metadata for {name} and update the configurations.\n")