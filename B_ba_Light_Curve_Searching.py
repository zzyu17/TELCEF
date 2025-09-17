from collections import defaultdict
import os

import lightkurve as lk
import matplotlib.pyplot as plt

from A_ab_Configuration_Loader import *




# Define the directories
lc_searching_dir = base_dir + config['directory']['lc_searching_dir']
os.makedirs(lc_searching_dir, exist_ok=True)
lc_search_result_dir = lc_searching_dir + config['directory']['lc_search_result_dir']
os.makedirs(lc_search_result_dir, exist_ok=True)
lc_search_result_path = lc_search_result_dir + f"/{name}_{mission}_Light_Curve_Search_Result.txt"
lc_search_result_file = open(lc_search_result_path, "w", encoding="utf-8")
lc_raw_plots_dir = lc_searching_dir + config['directory']['lc_raw_plots_dir']
os.makedirs(lc_raw_plots_dir, exist_ok=True)
lc_raw_plots_dir_source = lc_raw_plots_dir + f"/{name}"
os.makedirs(lc_raw_plots_dir_source, exist_ok=True)
lc_raw_plots_dir_source_mission_stage = lc_raw_plots_dir_source + f"/{mission}_{stage}-{sector}"
os.makedirs(lc_raw_plots_dir_source_mission_stage, exist_ok=True)




# Search for all the available light curves
lc_search_result = lk.search_lightcurve(name, mission=mission)
lc_search_result_file.write(str(lc_search_result))




# Download lightcurves of the specific mission, stage and supported author(s)
lc_search_result_mission_mask = np.char.find(lc_search_result.mission.astype(str), mission) >= 0 # define the mission mask
lc_search_result_stage_mask = np.char.find(lc_search_result[lc_search_result_mission_mask].mission.astype(str), str(sector)) >= 0 # define the stage mask
lc_search_result_supported_author_mask = np.zeros(len(lc_search_result), dtype=bool)
for author in config['lightkurve_search_lightcurve'][f'{mission}_supported_author']:
    lc_search_result_supported_author_mask |= np.char.find(lc_search_result.author.astype(str), author) >= 0 ##### define the author mask to add extra filtering criteria to include only the specific supported authors #####
lc_search_result_filtering_mask = lc_search_result_mission_mask & lc_search_result_stage_mask & lc_search_result_supported_author_mask
lc_search_result_mission_stage = lc_search_result[lc_search_result_filtering_mask]
lc_collection_mission_stage = lc_search_result_mission_stage.download_all(download_dir=lightkurve_cache_root_lc)
print(f"Successfully downloaded the {mission} {attribute} {sector} light curves of {name}.\n") # use "attribute" for lowercase stage name in the printed message


# Plot the single raw lightcurve
# group the lightcurves by author
authors_lc_mission_stage = [lc_mission_stage.author for lc_mission_stage in lc_collection_mission_stage]
authors_lc_mission_stage_dict = defaultdict(list)
for lc_mission_stage in lc_collection_mission_stage:
    authors_lc_mission_stage_dict[lc_mission_stage.author].append(lc_mission_stage)

lc_search_result_mission_stage_index_list = np.where(lc_search_result_filtering_mask)[0]
lc_search_result_mission_stage_author_list = [
    "Kepler" if mission == "K2" and lc_search_result_mission_stage_author == "K2" # replace "K2" with "Kepler" for K2 mission in search result
    else lc_search_result_mission_stage_author
    for lc_search_result_mission_stage_author in lc_search_result_mission_stage.author.astype(str)
]

for author_lc_mission_stage, lcs_mission_stage_author in authors_lc_mission_stage_dict.items():
    lc_mission_stage_author_index_list = lc_search_result_mission_stage_index_list[np.char.find(lc_search_result_mission_stage_author_list, author_lc_mission_stage) >= 0] # find the indices of search results whose author match that in authors_lc_mission_stage_dict
    print(f"Found {len(lcs_mission_stage_author)} light curve(s) for author {author_lc_mission_stage} in {mission} {attribute} {sector} in the search result table, matching at index " + ', '.join(str(index) for index in lc_mission_stage_author_index_list) + ".") # use "attribute" for lowercase stage name in the printed message

    for j in range(len(lcs_mission_stage_author)):
        lc_mission_stage_author = lcs_mission_stage_author[j]
        lc_mission_stage_author_plot, ax_lc_mission_stage_author = plt.subplots(figsize=(20, 5))
        lc_mission_stage_author.plot(ax=ax_lc_mission_stage_author, label=f"Estimated CDPP={lc_mission_stage_author.estimate_cdpp():.2f}")
        ax_lc_mission_stage_author.set_title(f"{name} {mission} {stage} {getattr(lc_mission_stage_author, attribute)} Author {lc_mission_stage_author.author} Raw Light Curve Exptime={lc_search_result.exptime.value[lc_mission_stage_author_index_list[j]]}s")
        ax_lc_mission_stage_author.legend(loc='upper right')
        lc_mission_stage_author_plot.figure.tight_layout()
        lc_mission_stage_author_plot.figure.savefig(lc_raw_plots_dir_source_mission_stage + f"/{name}_{mission}_{stage}-{getattr(lc_mission_stage_author, attribute)}_Author-{lc_mission_stage_author.author}_Raw_Light_Curve_Exptime={lc_search_result.exptime.value[lc_mission_stage_author_index_list[j]]}s.png")
        plt.close()
        print(f"Plotted the raw light curve for {mission} {attribute} {sector} author {lc_mission_stage_author.author} exptime={lc_search_result.exptime.value[lc_mission_stage_author_index_list[j]]}s.") # use "attribute" for lowercase stage name in the printed message
    print("")