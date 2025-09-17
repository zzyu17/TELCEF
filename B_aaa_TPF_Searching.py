from collections import defaultdict
import os

import lightkurve as lk
import matplotlib.pyplot as plt

from A_ab_Configuration_Loader import *




# Define the directories
tpf_searching_dir = base_dir + config['directory']['tpf_searching_dir']
os.makedirs(tpf_searching_dir, exist_ok=True)
tpf_search_result_dir = tpf_searching_dir + config['directory']['tpf_search_result_dir']
os.makedirs(tpf_search_result_dir, exist_ok=True)
tpf_search_result_path = tpf_search_result_dir + f"/{name}_{mission}_TPF_Search_Result.txt"
tpf_search_result_file = open(tpf_search_result_path, "w", encoding="utf-8")
tpf_plots_dir = tpf_searching_dir + config['directory']['tpf_plots_dir']
os.makedirs(tpf_plots_dir, exist_ok=True)
tpf_plots_dir_source = tpf_plots_dir + f"/{name}"
os.makedirs(tpf_plots_dir_source, exist_ok=True)
tpf_plots_dir_source_mission_stage = tpf_plots_dir_source + f"/{mission}_{stage}-{sector}"
os.makedirs(tpf_plots_dir_source_mission_stage, exist_ok=True)




# Search for all the available TPFs
tpf_search_result = lk.search_targetpixelfile(name, mission=mission)
tpf_search_result_file.write(str(tpf_search_result))




# Download TPFs of the specific mission and stage
tpf_search_result_mission_mask = np.char.find(tpf_search_result.mission.astype(str), mission) >= 0 # define the mission mask
tpf_search_result_stage_mask = np.char.find(tpf_search_result[tpf_search_result_mission_mask].mission.astype(str), str(sector)) >= 0 # define the stage mask
tpf_search_result_filtering_mask = tpf_search_result_mission_mask & tpf_search_result_stage_mask
tpf_search_result_mission_stage = tpf_search_result[tpf_search_result_filtering_mask]
tpf_collection_mission_stage = tpf_search_result_mission_stage.download_all(download_dir=lightkurve_cache_root_tpf)
print(f"Successfully downloaded the {mission} {attribute} {sector} TPFs of {name}.\n") # use "attribute" for lowercase stage name in the printed message


# Define the author
# when searching TPFs via Lightkurve, the author can only be "SPOC"
author = "SPOC" ##### set the author of the data to be processed #####


# Plot the single TPF
for j in range(len(tpf_collection_mission_stage)):
    tpf_mission_stage = tpf_collection_mission_stage[j]
    tpf_height = tpf_mission_stage.shape[1]
    tpf_width = tpf_mission_stage.shape[2]
    tpf_plot_cadence = config['tpf_plot']['cadence'][j] if len(config['tpf_plot']['cadence']) > 1 else config['tpf_plot']['cadence'][0] ##### set the cadence to be plotted #####
    tpf_mission_stage_plot, ax_tpf_mission_stage = plt.subplots(figsize=(10, 10))
    tpf_mission_stage[tpf_plot_cadence].plot(ax=ax_tpf_mission_stage)
    ax_tpf_mission_stage.set_title(f"{name} {mission} {stage} {getattr(tpf_mission_stage, attribute)} Author-{author} {tpf_width}x{tpf_height} TPF\n(Cadence {tpf_plot_cadence:04}) Exptime={tpf_search_result_mission_stage.exptime.value[j]}s")
    tpf_mission_stage_plot.figure.tight_layout()
    tpf_mission_stage_plot.figure.savefig(tpf_plots_dir_source_mission_stage + f"/{name}_{mission}_{stage}-{getattr(tpf_mission_stage, attribute)}_Author-{author}_{tpf_width}x{tpf_height}_TPF_(Cadence-{tpf_plot_cadence:04})_Exptime={tpf_search_result_mission_stage.exptime.value[j]}s.png")
    plt.close()
    print(f"Plotted the TPF (cadence {tpf_plot_cadence:04}) for {name} {mission} {attribute} {sector} exptime={tpf_search_result_mission_stage.exptime.value[j]}s.") # use "attribute" for lowercase stage name in the printed message
print("")