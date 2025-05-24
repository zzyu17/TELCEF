import os
from tess_stars2px import tess_stars2px_function_entry as tess_stars2px




# Set the single sector and exposure time to be processed
sector = 10 ##### set the single sector to be processed #####
exptime = 1800 ##### set the exposure time of data to be processed #####

# Define the starting point of TESS observation time
TESS_time = 2457000 # BJD

# ##### Define the target parameters #####
# name = "WASP-80"
# tic = 243921117
# coords = (303.16737208333325,-2.144218611111111)
# gaia = 4223507222112425344
# tess_mag = 10.3622

##### Define the target parameters #####
name = "WASP-107"
tic = 429302040
coords = (188.38685041666665, -10.146173611111111)
gaia = 3578638842054261248
tess_mag = 10.418

# Define the directories
root = os.getcwd()
eleanor_root = os.path.expanduser("~/.eleanor")
eleanor_root_metadata = eleanor_root + f"/metadata"
eleanor_root_metadata_sector = eleanor_root_metadata + f"/s00{sector}"
eleanor_root_tesscut = eleanor_root + f"/tesscut"
eleanor_root_tesscut_target = eleanor_root_tesscut + f"/{name}"
tess_stars2px_inspection_dir = root + "/00_02 tess_stars2px Inspection"
os.makedirs(tess_stars2px_inspection_dir, exist_ok=True)
tess_stars2px_inspection_fn = tess_stars2px_inspection_dir + f"/tess_stars2px_{name}_Sector {sector}.txt"




# Run tess_stars2px on the target
result = tess_stars2px(tic, coords[0], coords[1], sector)
tess_stars2px_inspection_file = open(tess_stars2px_inspection_fn, "w", encoding='utf-8')
for i in range(len(result)-1):
    tess_stars2px_inspection_file.write(f"{result[i]}\n")