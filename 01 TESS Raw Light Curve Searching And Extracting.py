import os
from collections import defaultdict

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np




### ------ Preparations ------ ###
##### Define the source parameters #####
name = "WASP-80"
tic = 243921117
coords = (303.16737208333325,-2.144218611111111)
gaia = 4223507222112425344
tess_mag = 10.3622

# ##### Define the source parameters #####
# name = "WASP-107"
# tic = 429302040
# coords = (188.38685041666665, -10.146173611111111)
# gaia = 3578638842054261248
# tess_mag = 10.418


##### Define the specific mission #####
mission = 'TESS'

# Define the attribute map and the stage map for different missions
attribute_map = {
    'Kepler': 'quarter',
    'K2': 'campaign',
    'TESS': 'sector'
}
attribute = attribute_map.get(mission)

stage_map = {
    'Kepler': 'Quarter',
    'K2': 'Campaign',
    'TESS': 'Sector'
}
stage = stage_map.get(mission)




# Define the directories
root = os.getcwd()
search_result_dir = root + "/01 Search Result"
os.makedirs(search_result_dir, exist_ok=True)
raw_lightcurve_plots_dir = root + "/01 Raw Light Curve Plots"
os.makedirs(raw_lightcurve_plots_dir, exist_ok=True)
raw_lightcurve_plots_parent_dir = raw_lightcurve_plots_dir + f"/{name}"
os.makedirs(raw_lightcurve_plots_parent_dir, exist_ok=True)
raw_lightcurve_plots_mission_parent_dir = raw_lightcurve_plots_parent_dir + f"/{mission}"
os.makedirs(raw_lightcurve_plots_mission_parent_dir, exist_ok=True)




### ------ Lightkurve ------ ###
# Search for all the available light curves
lc_search_result = lk.search_lightcurve(name)
lc_search_result_file = open(search_result_dir+f"/{name} Light Curve Search Result.txt", "w")
lc_search_result_file.write(str(lc_search_result))




# Download and plot light curves of the assigned mission
lc_collection_mission = lc_search_result[(np.char.find(lc_search_result.mission.astype(str), mission) >= 0)].download_all()
print(f"Downloaded the {mission} light curves.\n")


# Normalize and plot the raw lightcurve collection
lc_collection_mission_normalized_plot, ax_lc_collection_mission_normalized = plt.subplots(figsize=(20, 5))
for lc_mission in lc_collection_mission:
    lc_mission_normalized = lc_mission.normalize()
    lc_mission_normalized.plot(ax=ax_lc_collection_mission_normalized, label=f"{lc_mission_normalized.mission} {stage} {getattr(lc_mission_normalized, attribute)} Pipeline {lc_mission_normalized.author}, CDPP={lc_mission_normalized.estimate_cdpp():.2f}")
ax_lc_collection_mission_normalized.set_title(f"{name} {mission} Normalized Raw Light Curves")
lc_collection_mission_normalized_plot.figure.tight_layout()
lc_collection_mission_normalized_plot.figure.savefig(raw_lightcurve_plots_mission_parent_dir + f"/{name} {mission} Normalized Raw Light Curves.png")
print(f"Plotted the normalized {mission} raw light curve collection.\n")


# Plot the single raw lightcurve
# group the light curves by author
lc_mission_authors = [lc_mission.author for lc_mission in lc_collection_mission]
lc_mission_authors_dict = defaultdict(list)
for lc_mission in lc_collection_mission:
    lc_mission_authors_dict[lc_mission.author].append(lc_mission)

lc_search_result_author_list = [
    "Kepler" if mission == "K2" and lc_search_result_author == "K2" # replace "K2" with "Kepler" for K2 mission in search result
    else lc_search_result_author
    for lc_search_result_author in lc_search_result.author.astype(str)
]

for lc_mission_author, lcs_mission in lc_mission_authors_dict.items():
    lc_mission_author_index = np.where(np.char.find(lc_search_result_author_list, lc_mission_author) >= 0)[0][0] # find the index in the search result table where the author matches for the first time
    print(f"Found {len(lcs_mission)} light curve(s) for Pipeline {lc_mission_author} in the search result table, first of which matches at index {lc_mission_author_index}.")

    for j in range(len(lcs_mission)):
        lc_mission = lcs_mission[j]
        # print(list(lc_mission.meta.keys()))
        print(f"Plotting the single raw light curve for {mission} Pipeline {lc_mission.author} Exptime={lc_search_result.exptime.data[lc_mission_author_index + j]}s...")
        lc_mission_plot, ax_lc_mission = plt.subplots(figsize=(20, 5))
        lc_mission.plot(ax=ax_lc_mission, label=f"CDPP={lc_mission.estimate_cdpp():.2f}")
        ax_lc_mission.set_title(f"{name} {mission} {stage} {getattr(lc_mission, attribute)} Pipeline {lc_mission.author} Raw Light Curve Exptime={lc_search_result.exptime.data[lc_mission_author_index + j]}s")
        lc_mission_plot.figure.tight_layout()
        lc_mission_plot.figure.savefig(raw_lightcurve_plots_mission_parent_dir + f"/{name} {mission} {stage} {getattr(lc_mission, attribute)} Pipeline {lc_mission.author} Raw Light Curve Exptime={lc_search_result.exptime.data[lc_mission_author_index + j]}s.png")
    print("")